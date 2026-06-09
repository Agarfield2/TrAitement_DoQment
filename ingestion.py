from __future__ import annotations

import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# Structures de données

@dataclass
class BoundingBox:
    """Quadrilatère ICDAR : 4 sommets (x,y) sens horaire depuis haut-gauche."""
    x1: float; y1: float
    x2: float; y2: float
    x3: float; y3: float
    x4: float; y4: float

    @property
    def xmin(self) -> float: return min(self.x1, self.x2, self.x3, self.x4)
    @property
    def ymin(self) -> float: return min(self.y1, self.y2, self.y3, self.y4)
    @property
    def xmax(self) -> float: return max(self.x1, self.x2, self.x3, self.x4)
    @property
    def ymax(self) -> float: return max(self.y1, self.y2, self.y3, self.y4)


@dataclass
class TextLine:
    """Ligne de texte extraite (par OCR ou annotation SROIE)."""
    text: str
    bbox: Optional[BoundingBox]
    confidence: float = 1.0


@dataclass
class Passage:
    """Unité indexable — un bloc de texte avec ses métadonnées."""
    passage_id: str
    text: str
    source_file: str
    page_number: int
    bboxes: list[BoundingBox] = field(default_factory=list)
    avg_confidence: float = 1.0
    entities: Optional[dict] = None


# PDF → images


def rasterize_pdf(pdf_path: str | Path, dpi: int = 300) -> list[Image.Image]:
    try:
        from pdf2image import convert_from_path
        return convert_from_path(str(pdf_path), dpi=dpi)
    except ImportError:
        raise ImportError("Installer pdf2image : pip install pdf2image")


def load_image(image_path: str | Path) -> Image.Image:
    img = Image.open(str(image_path)).convert("RGB")
    return img


# Lecture des annotations SROIE
#
# NB : l'OCR réel (reconnaissance de texte sur image) ne vit PAS ici. Il est
# fourni par `doqment/ocr.py` (docTR par défaut, Tesseract en option), qui
# produit directement des `TextLine`. La classe ci-dessous se limite à lire
# la ground-truth ICDAR fournie avec le corpus SROIE.

class OCREngine:
    """Lecteur d'annotations SROIE (ground-truth ICDAR) → liste de TextLine.

    Pour l'OCR d'images sans annotation, utiliser `doqment/ocr.py`.
    """

    @staticmethod
    def from_sroie_annotation(annotation_path: str | Path) -> list[TextLine]:
        """
        Lit les annotations SROIE task1.
        Format : x1,y1,x2,y2,x3,y3,x4,y4,transcript
        """

        lines: list[TextLine] = []

        with open(annotation_path, encoding="utf-8", errors="ignore") as f:
            for raw in f:
                raw = raw.strip()

                if not raw:
                    continue

                parts = raw.split(",", 8)

                if len(parts) < 9:
                    continue

                coords = [float(p) for p in parts[:8]]
                transcript = parts[8].strip()

                bbox = BoundingBox(*coords)

                lines.append(TextLine(
                    text=transcript,
                    bbox=bbox,
                    confidence=1.0
                ))

        return lines


# Découpage en passages


def group_lines_into_passages(
    lines: list[TextLine],
    source_file: str,
    page_number: int = 1,
    max_chars: int = 400,
    y_gap_threshold: float = 20.0,
    entities: Optional[dict] = None,
) -> list[Passage]:

    if not lines:
        return []

    sorted_lines = sorted(
        [l for l in lines if l.bbox is not None],
        key=lambda l: l.bbox.ymin,
    )

    sorted_lines += [l for l in lines if l.bbox is None]

    passages: list[Passage] = []
    current_texts: list[str] = []
    current_bboxes: list[BoundingBox] = []
    current_confs: list[float] = []
    prev_ymax: Optional[float] = None
    passage_idx = 0

    def _flush():
        nonlocal passage_idx

        if not current_texts:
            return

        text = " ".join(current_texts).strip()

        if not text:
            return

        pid = f"{Path(source_file).stem}_p{page_number}_{passage_idx:03d}"

        passages.append(Passage(
            passage_id=pid,
            text=text,
            source_file=source_file,
            page_number=page_number,
            bboxes=list(current_bboxes),
            avg_confidence=float(np.mean(current_confs)) if current_confs else 1.0,
            entities=entities,
        ))

        passage_idx += 1
        current_texts.clear()
        current_bboxes.clear()
        current_confs.clear()

    for line in sorted_lines:

        if line.bbox is not None and prev_ymax is not None:
            gap = line.bbox.ymin - prev_ymax

            if gap > y_gap_threshold:
                _flush()

        if sum(len(t) for t in current_texts) + len(line.text) > max_chars:
            _flush()

        current_texts.append(line.text)

        if line.bbox:
            current_bboxes.append(line.bbox)
            prev_ymax = line.bbox.ymax

        current_confs.append(line.confidence)

    _flush()
    return passages


# Embeddings

class EmbeddingModel:

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    DIM = 768

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or self.MODEL_NAME
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "Installer sentence-transformers : pip install sentence-transformers"
                )

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:

        self._load()

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vectors.astype("float32")


# Indexation FAISS

class FAISSIndex:

    HNSW_M = 32
    HNSW_EF_SEARCH = 64

    def __init__(self, dim: int = EmbeddingModel.DIM):
        self.dim = dim
        self._index = None
        self._metadata: list[dict] = []

    def _build_index(self):
        try:
            import faiss

            self._index = faiss.IndexHNSWFlat(self.dim, self.HNSW_M)
            self._index.hnsw.efSearch = self.HNSW_EF_SEARCH

        except ImportError:
            raise ImportError("Installer FAISS : pip install faiss-cpu")

    def add(self, vectors: np.ndarray, passages: list[Passage]):

        if self._index is None:
            self._build_index()

        assert vectors.shape[0] == len(passages), "Vecteurs et passages désalignés"

        self._index.add(vectors)

        for p in passages:
            self._metadata.append({
                "passage_id": p.passage_id,
                "text": p.text,
                "source_file": p.source_file,
                "page_number": p.page_number,
                "avg_confidence": p.avg_confidence,
                "entities": p.entities,
                "bboxes": [asdict(b) for b in p.bboxes],
            })

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[dict]:

        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)

        qv = query_vector.reshape(1, -1).astype("float32")

        scores, indices = self._index.search(qv, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            meta = dict(self._metadata[idx])
            meta["score"] = float(score)
            results.append(meta)

        return results

    def save(self, directory: str | Path):

        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))

        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(self._metadata, f)

        print(f"Index sauvegardé : {self._index.ntotal} passages → {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSIndex":

        import faiss

        directory = Path(directory)

        obj = cls()

        obj._index = faiss.read_index(str(directory / "index.faiss"))

        with open(directory / "metadata.pkl", "rb") as f:
            obj._metadata = pickle.load(f)

        print(f"Index chargé : {obj._index.ntotal} passages depuis {directory}")

        return obj
