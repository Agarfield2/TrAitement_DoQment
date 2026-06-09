"""
Pipeline 2 — storage and encoding primitives.

Five things live here, because they are all I/O-bound and tightly
coupled to the on-disk layout :

- `detect_source_kind` / `rasterize_pdf` / `normalize_image` /
  `load_page_image`  — PDF & image preprocessing.
- `ColQwen2Encoder`  — the visual encoder (page side + query side).
- `QdrantStore`      — wrapper around a local Qdrant client.
- `MetadataStore`    — tiny SQLite store keyed by file MD5.
- `maxsim`           — late-interaction similarity for tests / fallback.
"""

import hashlib
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


### Rasterization ###

class SourceKind(str, Enum):
    """
    The three kinds of input documents the pipeline can ingest.
    """

    PDF_NATIVE = "pdf_native"
    PDF_SCANNED = "pdf_scanned"
    IMAGE = "image"


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def detect_source_kind(path, native_threshold=100):
    """
    Returns whether a path is a native PDF, scanned PDF or image.

    Args:
        path (Path): The candidate file path.
        native_threshold (int): Min extractable characters on page 1
            to be flagged PDF_NATIVE.

    Returns:
        SourceKind: One of PDF_NATIVE, PDF_SCANNED, IMAGE.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTS:
        return SourceKind.IMAGE
    if suffix != ".pdf":
        raise ValueError(f"Unsupported file format : {suffix}")

    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        if not pdf.pages:
            return SourceKind.PDF_SCANNED
        text = pdf.pages[0].extract_text() or ""
        return SourceKind.PDF_NATIVE if len(text.strip()) > native_threshold else SourceKind.PDF_SCANNED


def rasterize_pdf(pdf_path, output_dir, dpi=300, max_pages=None):
    """
    Rasterizes a PDF to one PNG per page under `output_dir`.

    Args:
        pdf_path (Path): Source PDF file.
        output_dir (Path): Folder to write the PNGs.
        dpi (int): Rasterization DPI.
        max_pages (int, optional): Cap on number of pages.

    Returns:
        list[Path]: Absolute paths of the produced PNG files, ordered.
    """

    from pdf2image import convert_from_path

    pdf_path, output_dir = Path(pdf_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=dpi)
    if max_pages:
        images = images[:max_pages]

    paths = []
    for i, img in enumerate(images, start=1):
        out = output_dir / f"{pdf_path.stem}_p{i:03d}.png"
        img.save(out, "PNG")
        paths.append(out)
    return paths


def normalize_image(image_path, output_dir):
    """
    Saves a standalone image as a normalized RGB PNG.

    Args:
        image_path (Path): Source image file.
        output_dir (Path): Folder to write the normalized PNG.

    Returns:
        Path: Path of the produced PNG file.
    """

    from PIL import Image, ImageOps

    image_path, output_dir = Path(image_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img = ImageOps.exif_transpose(Image.open(str(image_path))).convert("RGB")
    out = output_dir / f"{image_path.stem}_p001.png"
    img.save(out, "PNG")
    return out


def load_page_image(image_path):
    """
    Loads a rasterized page from disk into a PIL RGB image.

    Args:
        image_path (Path): The PNG file path.

    Returns:
        PIL.Image.Image: The loaded RGB image.
    """

    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(str(image_path))).convert("RGB")


def file_md5(path, chunk_size=65536):
    """
    Returns the MD5 hex digest of a file.

    Args:
        path (Path): The file to hash.
        chunk_size (int): Read buffer size.

    Returns:
        str: MD5 hex digest.
    """

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


### Encoder ###

class ColQwen2Encoder:
    """
    Unified ColQwen2 encoder for pages and queries.

    Loads ColQwen2 once ; encode_pages / encode_query reuse the same
    weights. Embeddings are moved to CPU before being returned so the
    GPU is free for the next call.
    """

    def __init__(self, model_name="vidore/colqwen2-v1.0",
                 device="cuda:0", dtype="bfloat16"):
        """
        Loads ColQwen2 weights and processor.

        Args:
            model_name (str): HuggingFace model id.
            device (str): "cuda:0", "cpu", or any torch device.
            dtype (str): Compute dtype name (bfloat16/float16/float32).
        """

        from colpali_engine.models import ColQwen2, ColQwen2Processor
        import torch

        attn_impl = None
        try:
            from transformers.utils.import_utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                attn_impl = "flash_attention_2"
        except Exception:
            pass

        logger.info("Loading ColQwen2 : %s (%s, %s)", model_name, device, dtype)
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            attn_implementation=attn_impl,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        self.device = device

    def encode_pages(self, images, batch_size=4):
        """
        Encodes page-images into per-image multi-vector tensors.

        Args:
            images (list[PIL.Image.Image]): The pages to encode.
            batch_size (int): How many pages per forward pass.

        Returns:
            list[torch.Tensor]: One tensor per image, shape
                (n_patches_i, 128), on CPU.
        """

        import torch

        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor.process_images(batch).to(self.device)
            with torch.no_grad():
                emb = self.model(**inputs)
            masks = inputs.get("attention_mask")
            for j, vecs in enumerate(emb):
                if masks is not None:
                    vecs = vecs[masks[j].bool()]
                out.append(vecs.cpu())
        return out

    def encode_query(self, query):
        """
        Encodes a textual query into a multi-vector representation.

        Args:
            query (str): The user question.

        Returns:
            torch.Tensor: Shape (n_query_tokens, 128), on CPU.
        """

        import torch

        inputs = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs)
        masks = inputs.get("attention_mask")
        vecs = emb[0]
        if masks is not None:
            vecs = vecs[masks[0].bool()]
        return vecs.cpu()


def maxsim(query_emb, page_emb):
    """
    Computes the late-interaction MaxSim score between query and page.

    Args:
        query_emb (torch.Tensor): Shape (n_query_tokens, dim).
        page_emb (torch.Tensor): Shape (n_patches, dim).

    Returns:
        float: The MaxSim score (sum over query of max over page).
    """

    sims = query_emb.float() @ page_emb.float().T
    return float(sims.max(dim=1).values.sum().item())


### Qdrant store ###

class QdrantStore:
    """
    Wraps a local Qdrant client for multi-vector page embeddings.

    A page is one Qdrant point ; its `vector` is the list of patch
    vectors (Qdrant computes MaxSim natively when configured with
    `MultiVectorConfig(comparator=MAX_SIM)`).
    """

    COLLECTION = "pages"
    VECTOR_DIM = 128

    def __init__(self, qdrant_dir):
        """
        Opens (or creates) a Qdrant database at the given path.

        Args:
            qdrant_dir (Path): On-disk location of the Qdrant database.
        """

        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        qdrant_dir = Path(qdrant_dir)
        qdrant_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client = QdrantClient(path=str(qdrant_dir))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Qdrant local '{qdrant_dir}' est deja verrouille. "
                f"Causes possibles : un autre process l'utilise (autre onglet/"
                f"instance Streamlit, `scripts/phase2.py`, notebook...), ou un "
                f"verrou perime apres un arret brutal. Solution : fermer les "
                f"autres process, ou — si aucun n'est actif — supprimer "
                f"'{qdrant_dir}/.lock', puis reessayer."
            ) from exc
        self._qmodels = qmodels

        if not self.client.collection_exists(self.COLLECTION):
            self.client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=self.VECTOR_DIM,
                    distance=qmodels.Distance.COSINE,
                    multivector_config=qmodels.MultiVectorConfig(
                        comparator=qmodels.MultiVectorComparator.MAX_SIM,
                    ),
                ),
            )

    def upsert_page(self, page_id, vectors, payload):
        """
        Inserts or updates a page point.

        Args:
            page_id (uuid.UUID): The page identifier.
            vectors (list[list[float]]): The patch-level vectors.
            payload (dict): Metadata to store alongside.
        """

        point = self._qmodels.PointStruct(
            id=str(page_id), vector=vectors, payload=payload,
        )
        self.client.upsert(collection_name=self.COLLECTION, points=[point])

    def search(self, query_vectors, top_k):
        """
        Runs a MaxSim search for the closest pages.

        Args:
            query_vectors (list[list[float]]): The query multi-vector.
            top_k (int): Max number of pages to return.

        Returns:
            list[dict]: One result per hit, each with score+payload+id.
        """

        hits = self.client.query_points(
            collection_name=self.COLLECTION,
            query=query_vectors,
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {"id": h.id, "score": float(h.score), "payload": h.payload}
            for h in hits
        ]

    def close(self):
        """
        Closes the underlying Qdrant client.
        """

        try:
            self.client.close()
        except Exception:
            pass


### SQLite metadata ###

class MetadataStore:
    """
    Tiny SQLite store keyed by file MD5, for ingestion idempotence.
    """

    def __init__(self, db_path):
        """
        Opens (or creates) the SQLite database.

        Args:
            db_path (Path): Path to the SQLite file.
        """

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            " doc_id TEXT PRIMARY KEY, source_path TEXT, md5 TEXT UNIQUE,"
            " n_pages INTEGER, source_kind TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS pages ("
            " page_id TEXT PRIMARY KEY, doc_id TEXT, page_num INTEGER,"
            " image_path TEXT, dpi INTEGER)"
        )
        self.conn.commit()

    def already_ingested(self, md5):
        """
        Returns True if a document with that MD5 is already indexed.

        Args:
            md5 (str): The file's MD5 hex digest.

        Returns:
            bool: True if a doc row exists.
        """

        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE md5 = ?", (md5,),
        ).fetchone()
        return row is not None

    def add_document(self, doc_id, source_path, md5, n_pages, source_kind):
        """
        Inserts a document row.
        """

        self.conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)",
            (doc_id, str(source_path), md5, n_pages, source_kind),
        )
        self.conn.commit()

    def add_page(self, page_id, doc_id, page_num, image_path, dpi):
        """
        Inserts a page row.
        """

        self.conn.execute(
            "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?)",
            (str(page_id), doc_id, page_num, str(image_path), dpi),
        )
        self.conn.commit()

    def close(self):
        """
        Closes the database connection.
        """

        try:
            self.conn.close()
        except Exception:
            pass


### Context manager ###

@contextmanager
def open_stores(settings):
    """
    Opens the Qdrant + SQLite stores and closes them at exit.

    The Qdrant client is closed even if the metadata store or the body
    raises — otherwise a leaked local client would keep the storage folder
    locked and make every later open fail ("already accessed by another
    instance").

    Args:
        settings (Settings): Project settings.

    Yields:
        tuple: (QdrantStore, MetadataStore).
    """

    vec = QdrantStore(settings.phase2_qdrant_dir)
    try:
        meta = MetadataStore(settings.phase2_metadata_db)
        try:
            yield vec, meta
        finally:
            meta.close()
    finally:
        vec.close()
