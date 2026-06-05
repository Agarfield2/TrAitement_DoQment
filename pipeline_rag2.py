"""
INSTALLATION
  pip install -r Requirements.txt
  #   CPU  : pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  #   CUDA : pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

  # Modele Mistral-7B quantise (GGUF) ~4 Go :
  # https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
  # Fichier recommande : mistral-7b-instruct-v0.2.Q4_K_M.gguf

USAGE
  # Etape 1 : ingerer les documents (OCR + index)
  python pipeline_rag.py ingest \\
      --docs  data/task1et2_test \\
      --annot "data/text.task1et2-test)" \\
      --index data/faiss_index

  # Etape 2 : poser une question
  python pipeline_rag.py query \\
      --index  data/faiss_index \\
      --model  models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \\
      --question "What is the total amount on the receipt?"

  # Mode interactif
  python pipeline_rag.py repl \\
      --index data/faiss_index \\
      --model models/mistral-7b-instruct-v0.2.Q4_K_M.gguf

  # Recherche seule (sans LLM, pour tester FAISS)
  python pipeline_rag.py search \\
      --index data/faiss_index \\
      --question "total amount"
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm


# Structures de donnees

@dataclass
class BoundingBox:
    x1: float; y1: float
    x2: float; y2: float
    x3: float; y3: float
    x4: float; y4: float

    @property
    def ymin(self) -> float: return min(self.y1, self.y2, self.y3, self.y4)
    @property
    def ymax(self) -> float: return max(self.y1, self.y2, self.y3, self.y4)


@dataclass
class TextLine:
    text: str
    bbox: Optional[BoundingBox]
    confidence: float = 1.0


@dataclass
class Passage:
    passage_id: str
    text: str
    source_file: str
    page_number: int
    bboxes: list[BoundingBox] = field(default_factory=list)
    avg_confidence: float = 1.0
    entities: Optional[dict] = None



# ETAPE 1 - OCR (Tesseract)


class TesseractOCREngine:
    """OCR engine basé sur Tesseract — remplace PaddleOCR (incompatible Python ≥ 3.10+).

    Utilise le chemin Tesseract défini par la variable d'environnement
    ``DOQMENT_TESSERACT_PATH`` (défaut : ``/usr/bin/tesseract``).
    """

    LOW_CONTRAST_RMS = 20.0

    def __init__(self, lang: str = "fra+eng", use_gpu: bool = False):
        # use_gpu conservé pour compatibilité d'interface — Tesseract est CPU-only.
        self.lang = lang
        import os
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = os.environ.get(
            "DOQMENT_TESSERACT_PATH", "/usr/bin/tesseract"
        )
        self._pytesseract = pytesseract

    def _preprocess(self, img: Image.Image) -> Image.Image:
        """Contrast adaptatif → seuillage → dilatation."""
        try:
            import cv2
        except (ImportError, AttributeError):
            return img.convert("L")
        arr = np.array(img.convert("L"), dtype=float)
        if float(arr.std()) < self.LOW_CONTRAST_RMS:
            import cv2 as _cv2
            bgr = _cv2.cvtColor(np.array(img.convert("RGB")), _cv2.COLOR_RGB2BGR)
            bgr = _cv2.convertScaleAbs(bgr, alpha=1.5, beta=0)
            img = Image.fromarray(_cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB))
        arr8 = np.array(img.convert("L"))
        binary = cv2.adaptiveThreshold(
            arr8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return Image.fromarray(cv2.dilate(binary, kernel, iterations=1))

    def run(self, img: Image.Image) -> list[TextLine]:
        processed = self._preprocess(img)
        try:
            available = self._pytesseract.get_languages()
            parts = [p for p in self.lang.split("+") if p in available]
            lang = "+".join(parts) if parts else (available[0] if available else "eng")
        except Exception:
            lang = self.lang

        data = self._pytesseract.image_to_data(
            processed, lang=lang,
            config="--oem 3 --psm 3",
            output_type=self._pytesseract.Output.DICT,
        )
        by_line: dict = {}
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            by_line.setdefault(key, []).append(i)

        lines: list[TextLine] = []
        for key in sorted(by_line.keys()):
            idxs = by_line[key]
            words = [data["text"][i].strip() for i in idxs]
            text = " ".join(w for w in words if w)
            if not text:
                continue
            xs = [int(data["left"][i]) for i in idxs]
            ys = [int(data["top"][i]) for i in idxs]
            rs = [int(data["left"][i]) + int(data["width"][i]) for i in idxs]
            bs = [int(data["top"][i]) + int(data["height"][i]) for i in idxs]
            x1, y1, x2, y2 = min(xs), min(ys), max(rs), max(bs)
            confs = []
            for i in idxs:
                try:
                    c = float(data["conf"][i])
                    if c >= 0:
                        confs.append(c / 100.0)
                except (TypeError, ValueError):
                    pass
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            bbox = BoundingBox(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y1),
                x3=float(x2), y3=float(y2),
                x4=float(x1), y4=float(y2),
            )
            lines.append(TextLine(text=text, bbox=bbox, confidence=avg_conf))
        return lines

    @staticmethod
    def from_sroie_annotation(path: str | Path) -> list[TextLine]:
        
        lines: list[TextLine] = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split(",", 8)
                if len(parts) < 9:
                    continue
                coords = [float(p) for p in parts[:8]]
                text = parts[8].strip()
                if text:
                    lines.append(TextLine(
                        text=text,
                        bbox=BoundingBox(*coords),
                        confidence=1.0
                    ))
        return lines


def load_image_or_pdf(path: Path, dpi: int = 300) -> list[Image.Image]:
    
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        imgs = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            imgs.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
        return imgs
    return [Image.open(str(path)).convert("RGB")]



# ETAPE 2 - Regroupement en passages


def group_lines_into_passages(
    lines: list[TextLine],
    source_file: str,
    page_number: int = 1,
    max_chars: int = 1200,
    y_gap_threshold: float = 60.0,
    entities: Optional[dict] = None,
) -> list[Passage]:
    
    if not lines:
        return []

    sorted_lines = sorted(
        [l for l in lines if l.bbox is not None],
        key=lambda l: l.bbox.ymin,
    ) + [l for l in lines if l.bbox is None]

    passages: list[Passage] = []
    cur_texts, cur_bboxes, cur_confs = [], [], []
    prev_ymax: Optional[float] = None
    idx = 0

    def flush():
        nonlocal idx
        if not cur_texts:
            return
        text = " ".join(cur_texts).strip()
        if not text:
            return
        pid = f"{Path(source_file).stem}_p{page_number}_{idx:03d}"
        passages.append(Passage(
            passage_id=pid,
            text=text,
            source_file=source_file,
            page_number=page_number,
            bboxes=list(cur_bboxes),
            avg_confidence=float(np.mean(cur_confs)) if cur_confs else 1.0,
            entities=entities,
        ))
        idx += 1
        cur_texts.clear(); cur_bboxes.clear(); cur_confs.clear()

    for line in sorted_lines:
        if line.bbox is not None and prev_ymax is not None:
            if line.bbox.ymin - prev_ymax > y_gap_threshold:
                flush()
        if sum(len(t) for t in cur_texts) + len(line.text) > max_chars:
            flush()
        cur_texts.append(line.text)
        if line.bbox:
            cur_bboxes.append(line.bbox)
            prev_ymax = line.bbox.ymax
        cur_confs.append(line.confidence)

    flush()
    return passages



# ETAPE 3 - Embeddings (SentenceTransformer)


class EmbeddingModel:
    
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    DIM = 768

    def __init__(self, model_name: Optional[str] = None):
        self._name = model_name or self.MODEL_NAME
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"  [Embedding] Chargement : {self._name}")
                self._model = SentenceTransformer(self._name)
            except ImportError:
                raise ImportError("pip install sentence-transformers")

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        self._load()
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")



# ETAPE 4 - Index FAISS (HNSW)


class FAISSIndex:
    
    HNSW_M = 32
    HNSW_EF_SEARCH = 64

    def __init__(self, dim: int = EmbeddingModel.DIM):
        self.dim = dim
        self._index = None
        self._metadata: list[dict] = []

    def _init(self):
        try:
            import faiss
            self._index = faiss.IndexHNSWFlat(self.dim, self.HNSW_M)
            self._index.hnsw.efSearch = self.HNSW_EF_SEARCH
        except ImportError:
            raise ImportError("pip install faiss-cpu")

    def add(self, vectors: np.ndarray, passages: list[Passage]):
        if self._index is None:
            self._init()
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
        scores, indices = self._index.search(
            query_vector.reshape(1, -1).astype("float32"), k
        )
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
        print(f"  [FAISS] {self._index.ntotal} passages -> {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSIndex":
        import faiss
        directory = Path(directory)
        obj = cls()
        obj._index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "rb") as f:
            obj._metadata = pickle.load(f)
        print(f"  [FAISS] {obj._index.ntotal} passages charges depuis {directory}")
        return obj



# ETAPE 5 - LLM Mistral-7B (llama-cpp, 100% local)


class MistralLLM:
    

    PROMPT_TEMPLATE = (
        "<s>[INST] You are a receipt and invoice analysis assistant. "
        "The context below contains text extracted from MULTIPLE different receipts/invoices. "
        "Each source block is labeled with its filename. "
        "Answer the question using ONLY the context below. "
        "Always cite the source filename in your answer. "
        "If multiple receipts match, list each one separately. "
        "If the answer is not found, say \'Not found in provided receipts\'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question} [/INST]"
    )

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 32768,
        n_gpu_layers: int = 0,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def close(self):
        """Fermeture propre du modele pour eviter l exception __del__."""
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None

    def _load(self):
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError("pip install llama-cpp-python")
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Modele introuvable : {self.model_path}\n"
                    "Telechargez : https://huggingface.co/TheBloke/"
                    "Mistral-7B-Instruct-v0.2-GGUF\n"
                    "Fichier recommande : mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                )
            print(f"  [LLM] Chargement Mistral depuis {self.model_path} ...")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            print("  [LLM] Modele pret")

    def generate(self, question: str, passages: list[dict]) -> str:
        
        self._load()

        # Construire le contexte depuis les top-k passages
        context_parts = []
        for i, p in enumerate(passages, 1):
            src = Path(p["source_file"]).name
            context_parts.append(
                f"[Source {i}: {src} - page {p['page_number']} "
                f"(score {p['score']:.3f})]\n{p['text']}"
            )
        context = "\n\n".join(context_parts)

        prompt = self.PROMPT_TEMPLATE.format(context=context, question=question)

        t0 = time.time()
        response = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            echo=False,
            stop=["</s>", "[INST]"],
        )
        elapsed = time.time() - t0

        answer = response["choices"][0]["text"].strip()
        tokens = response["usage"]["total_tokens"]
        print(f"  [LLM] {tokens} tokens generes en {elapsed:.1f}s")
        return answer



# Pipeline RAG - orchestrateur


class RAGPipeline:
    

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"}

    def __init__(
        self,
        index_dir: str = "data/faiss_index",
        ocr_lang: str = "en",
        use_gpu_ocr: bool = False,
        embed_model: Optional[str] = None,
        llm_path: Optional[str] = None,
        n_gpu_layers: int = 0,
    ):
        self.index_dir = Path(index_dir)
        self.ocr = TesseractOCREngine(lang=ocr_lang, use_gpu=use_gpu_ocr)
        self.embedder = EmbeddingModel(embed_model)
        self.faiss = FAISSIndex()
        self.llm = MistralLLM(llm_path, n_gpu_layers=n_gpu_layers) if llm_path else None

    #  Ingestion 

    def ingest(
        self,
        docs_dir: str | Path,
        annot_dir: Optional[str | Path] = None,
        max_docs: Optional[int] = None,
    ):
        
        docs_dir = Path(docs_dir)
        annot_dir = Path(annot_dir) if annot_dir else None

        files = sorted([
            f for f in docs_dir.iterdir()
            if f.suffix.lower() in self.IMG_EXTS
        ])
        if max_docs:
            files = files[:max_docs]

        print(f"\n{'='*60}")
        print(f"  INGESTION - {len(files)} document(s)")
        print(f"  Source      : {docs_dir}")
        print(f"  Annotations : {annot_dir or 'Tesseract direct'}")
        print(f"{'='*60}")

        all_passages: list[Passage] = []

        for img_path in tqdm(files, desc="OCR + passages"):
            ann_path = annot_dir / f"{img_path.stem}.txt" if annot_dir else None

            # Annotations SROIE disponibles -> pas besoin d'OCR
            if ann_path and ann_path.exists():
                lines = TesseractOCREngine.from_sroie_annotation(ann_path)
            else:
                # PaddleOCR sur l'image (ou chaque page du PDF)
                images = load_image_or_pdf(img_path)
                lines = []
                for page_idx, img in enumerate(images):
                    lines.extend(self.ocr.run(img))

            passages = group_lines_into_passages(
                lines=lines,
                source_file=str(img_path),
            )
            all_passages.extend(passages)

        if not all_passages:
            print("Aucun passage extrait. Verifiez les chemins.")
            return

        print(f"\n  {len(all_passages)} passages extraits")

        print(f"\n  Calcul des embeddings...")
        texts = [p.text for p in all_passages]
        vectors = self.embedder.encode(texts, show_progress=True)

        print(f"\n  Construction index FAISS...")
        self.faiss.add(vectors, all_passages)
        self.faiss.save(self.index_dir)

        stats = {
            "total_passages": len(all_passages),
            "total_documents": len(files),
            "avg_confidence": float(np.mean([p.avg_confidence for p in all_passages])),
        }
        with open(self.index_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n  Ingestion terminee : {stats}")

    #  Chargement index existant 

    def load_index(self):
        self.faiss = FAISSIndex.load(self.index_dir)

    #  Requete RAG 

    def query(self, question: str, k: int = 5) -> dict:
        
        if self.faiss._index is None or self.faiss._index.ntotal == 0:
            raise RuntimeError("Index vide - lancez d'abord ingest()")
        if self.llm is None:
            raise RuntimeError("LLM non configure (--model manquant)")

        q_vec = self.embedder.encode([question])
        passages = self.faiss.search(q_vec[0], k=k)
        answer = self.llm.generate(question, passages)

        return {"question": question, "answer": answer, "passages": passages}

    def search_only(self, question: str, k: int = 5) -> list[dict]:
        
        q_vec = self.embedder.encode([question])
        return self.faiss.search(q_vec[0], k=k)

    #  Affichage 

    @staticmethod
    def print_result(result: dict):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  QUESTION : {result['question']}")
        print(sep)
        print(f"\n  REPONSE MISTRAL :\n  {result['answer']}")
        print(f"\n  PASSAGES SOURCES (top {len(result['passages'])}) :")
        for i, p in enumerate(result["passages"], 1):
            src = Path(p["source_file"]).name
            print(f"  [{i}] score={p['score']:.4f}  {src}  page {p['page_number']}")
            print(f"       {p['text'][:120]}")
        print(sep)



# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline RAG : Tesseract -> FAISS -> Mistral-7B (100% local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    pi = sub.add_parser("ingest", help="OCR + embeddings + index FAISS")
    pi.add_argument("--docs",     required=True,            help="Dossier images/PDFs")
    pi.add_argument("--annot",    default=None,             help="Dossier annotations SROIE (optionnel)")
    pi.add_argument("--index",    default="data/faiss_index")
    pi.add_argument("--lang",     default="fra+eng",         help="Langue Tesseract (fra+eng, eng, ...)")
    pi.add_argument("--gpu-ocr",  action="store_true",      help="Ignoré (Tesseract est CPU-only)")
    pi.add_argument("--max-docs", type=int, default=None)

    # query
    pq = sub.add_parser("query", help="Question -> recherche + Mistral")
    pq.add_argument("--index",      default="data/faiss_index")
    pq.add_argument("--model",      required=True,  help="Chemin .gguf Mistral")
    pq.add_argument("--question",   required=True)
    pq.add_argument("--k",          type=int, default=5)
    pq.add_argument("--gpu-layers", type=int, default=0,   help="Couches GPU llama-cpp")
    pq.add_argument("--lang",       default="en")

    # search (sans LLM)
    ps = sub.add_parser("search", help="Recherche FAISS seule (sans LLM)")
    ps.add_argument("--index",    default="data/faiss_index")
    ps.add_argument("--question", required=True)
    ps.add_argument("--k",        type=int, default=5)
    ps.add_argument("--lang",     default="en")

    # repl
    pr = sub.add_parser("repl", help="Mode interactif")
    pr.add_argument("--index",      default="data/faiss_index")
    pr.add_argument("--model",      required=True)
    pr.add_argument("--k",          type=int, default=5)
    pr.add_argument("--gpu-layers", type=int, default=0)
    pr.add_argument("--lang",       default="en")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "ingest":
        pipeline = RAGPipeline(
            index_dir=args.index,
            ocr_lang=args.lang,
            use_gpu_ocr=args.gpu_ocr,
        )
        pipeline.ingest(
            docs_dir=args.docs,
            annot_dir=args.annot,
            max_docs=args.max_docs,
        )

    elif args.cmd == "query":
        pipeline = RAGPipeline(
            index_dir=args.index,
            ocr_lang=args.lang,
            llm_path=args.model,
            n_gpu_layers=args.gpu_layers,
        )
        pipeline.load_index()
        try:
            result = pipeline.query(args.question, k=args.k)
            RAGPipeline.print_result(result)
        finally:
            if pipeline.llm:
                pipeline.llm.close()

    elif args.cmd == "search":
        pipeline = RAGPipeline(index_dir=args.index, ocr_lang=args.lang)
        pipeline.load_index()
        passages = pipeline.search_only(args.question, k=args.k)
        print(f"\nTop-{args.k} pour : '{args.question}'")
        for i, p in enumerate(passages, 1):
            src = Path(p["source_file"]).name
            print(f"\n[{i}] score={p['score']:.4f}  {src}")
            print(f"     {p['text']}")

    elif args.cmd == "repl":
        pipeline = RAGPipeline(
            index_dir=args.index,
            ocr_lang=args.lang,
            llm_path=args.model,
            n_gpu_layers=args.gpu_layers,
        )
        pipeline.load_index()
        print("\nMode interactif - 'exit' pour quitter")
        try:
            while True:
                try:
                    question = input("\nQuestion > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nAu revoir.")
                    break
                if question.lower() in ("exit", "quit", "q"):
                    break
                if not question:
                    continue
                result = pipeline.query(question, k=args.k)
                RAGPipeline.print_result(result)
        finally:
            if pipeline.llm:
                pipeline.llm.close()


if __name__ == "__main__":
    main()
