from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


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
    def ymax(self) -> float: return max(self.x1, self.y2, self.y3, self.y4)


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


# OCR

class OCREngine:
    """Wrapper PaddleOCR avec fallback sur annotations SROIE (ground-truth)."""

    def __init__(self, use_gpu: bool = False, lang: str = "en"):
        self._ocr = None
        self._use_gpu = use_gpu
        self._lang = lang

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                self._model = SentenceTransformer(
                    self._model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )

            except Exception as e:
                raise RuntimeError(
                    f"Impossible de charger le modèle SentenceTransformer "
                    f"'{self._model_name}'.\n"
                    f"Vérifie la connexion internet ou télécharge le modèle manuellement.\n"
                    f"Erreur originale : {e}"
                )

    def run(self, image: Image.Image) -> list[TextLine]:
        self._load()
        import numpy as np

        result = self._ocr.ocr(np.array(image), cls=True)
        lines: list[TextLine] = []

        if result and result[0]:
            for item in result[0]:
                pts, (text, conf) = item

                bbox = BoundingBox(
                    x1=pts[0][0], y1=pts[0][1],
                    x2=pts[1][0], y2=pts[1][1],
                    x3=pts[2][0], y3=pts[2][1],
                    x4=pts[3][0], y4=pts[3][1],
                )

                lines.append(TextLine(
                    text=text,
                    bbox=bbox,
                    confidence=float(conf)
                ))

        return lines

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


# Pipeline d'ingestion

class IngestionPipeline:

    def __init__(
        self,
        index_dir: str = "data/processed",
        use_paddle_ocr: bool = False,
        use_gpu: bool = False,
        embed_model: Optional[str] = None,
    ):
        self.index_dir = Path(index_dir)
        self.use_paddle_ocr = use_paddle_ocr
        self.ocr_engine = OCREngine(use_gpu=use_gpu) if use_paddle_ocr else None
        self.embed_model = EmbeddingModel(embed_model)
        self.faiss_index = FAISSIndex()
        self._all_passages: list[Passage] = []

    def process_image(
        self,
        image_path: str | Path,
        annotation_path: Optional[str | Path] = None,
        entities_path: Optional[str | Path] = None,
    ) -> list[Passage]:

        image_path = Path(image_path)

        if annotation_path and Path(annotation_path).exists() and not self.use_paddle_ocr:
            lines = OCREngine.from_sroie_annotation(annotation_path)

        elif self.use_paddle_ocr:
            img = load_image(image_path)
            lines = self.ocr_engine.run(img)

        else:
            print(f"  [SKIP] {image_path.name} : pas d'annotation et OCR désactivé")
            return []

        entities = None

        if entities_path and Path(entities_path).exists():
            with open(entities_path, encoding="utf-8", errors="ignore") as f:
                try:
                    entities = json.load(f)
                except json.JSONDecodeError:
                    pass

        passages = group_lines_into_passages(
            lines=lines,
            source_file=str(image_path),
            entities=entities,
        )

        return passages

    def ingest_directory(
        self,
        task1_dir: str | Path,
        task2_dir: Optional[str | Path] = None,
        image_extensions: tuple = (".jpg", ".jpeg", ".png"),
        max_docs: Optional[int] = None,
    ):

        task1_dir = Path(task1_dir)
        task2_dir = Path(task2_dir) if task2_dir else None

        image_files = sorted([
            f for f in task1_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        if max_docs:
            image_files = image_files[:max_docs]

        print(f"\n── Ingestion : {len(image_files)} documents ──────────────────")

        all_texts: list[str] = []
        all_passages: list[Passage] = []

        for img_path in tqdm(image_files, desc="OCR + passages"):
            stem = img_path.stem

            ann_path = task1_dir / f"{stem}.txt"

            # SROIE-Dataset_v2 : images dans img/, annotation box/ en
            # dossier frere. Si elle n'est pas a cote de l'image, la
            # chercher dans ../box/.
            if not ann_path.exists():
                sibling = task1_dir.parent / "box" / f"{stem}.txt"
                if sibling.exists():
                    ann_path = sibling

            ent_path = None
            if task2_dir:
                candidate = task2_dir / f"{stem}.txt"
                if candidate.exists():
                    ent_path = candidate

            passages = self.process_image(
                image_path=img_path,
                annotation_path=ann_path,
                entities_path=ent_path,
            )

            all_passages.extend(passages)
            all_texts.extend(p.text for p in passages)

        print(f"\n── Embeddings : {len(all_passages)} passages ──────────────────")

        if not all_passages:
            print("Aucun passage trouvé. Vérifier les chemins et annotations.")
            return

        vectors = self.embed_model.encode(all_texts, show_progress=True)

        print(f"\n── Indexation FAISS ──────────────────────────────────────────")

        self.faiss_index.add(vectors, all_passages)
        self._all_passages.extend(all_passages)

        self.faiss_index.save(self.index_dir)

        print(f"\n✓ Ingestion terminée : {len(all_passages)} passages indexés")

    def stats(self) -> dict:

        if not self._all_passages:
            return {}

        texts = [p.text for p in self._all_passages]
        lengths = [len(t) for t in texts]
        docs = len({p.source_file for p in self._all_passages})

        return {
            "total_passages": len(self._all_passages),
            "total_documents": docs,
            "avg_passage_len": float(np.mean(lengths)),
            "min_passage_len": int(min(lengths)),
            "max_passage_len": int(max(lengths)),
            "avg_confidence": float(np.mean([p.avg_confidence for p in self._all_passages])),
        }


# Script standalone

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Pipeline d'ingestion Phase 1")

    parser.add_argument(
        "--task1",
        default="data/SROIE-Dataset_v2/train/img"
    )

    parser.add_argument(
        "--task2",
        default="data/SROIE-Dataset_v2/train/entities"
    )

    parser.add_argument(
        "--index-dir",
        default="data/processed"
    )

    parser.add_argument(
        "--paddle",
        action="store_true",
        help="Utiliser PaddleOCR au lieu des annotations ground-truth"
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None
    )

    args = parser.parse_args()

    pipeline = IngestionPipeline(
        index_dir=args.index_dir,
        use_paddle_ocr=args.paddle,
    )

    pipeline.ingest_directory(
        task1_dir=args.task1,
        task2_dir=args.task2,
        max_docs=args.max_docs,
    )

    print("\nStatistiques :", pipeline.stats())
