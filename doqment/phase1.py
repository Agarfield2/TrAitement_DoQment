"""
Pipeline 1 — textual RAG.

Three entry points :

- `ingest_directory(...)`        builds the FAISS index from a folder.
- `ask_document(file, question)` answers a question about ONE document.
- `ask_database(question)`       answers a question across the whole index.

The heavy lifting (OCR, chunking, embeddings, FAISS) lives in the
canonical `ingestion.py` file. We never modify that file — we wrap it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import ingestion as _ing
from doqment import llm as _llm
from doqment import ocr as _ocr
from doqment.settings import load_settings


logger = logging.getLogger(__name__)


### Public Types ###

@dataclass(frozen=True)
class Source:
    """
    A passage cited in an answer.

    Attributes:
        document: Name of the source file (basename, no path).
        page: 1-based page number inside the document.
        score: Similarity score returned by the index.
        snippet: A short excerpt of the passage text.
        entities: Optional SROIE-style entities dict.
    """

    document: str
    page: int
    score: float
    snippet: str
    entities: dict = None


@dataclass(frozen=True)
class Answer:
    """
    The full result of a question.

    Attributes:
        text: The LLM's answer.
        sources: The retrieved passages cited.
    """

    text: str
    sources: list


_REFUSAL = "I do not have this information in the provided documents."


### Ingestion ###

def ingest_directory(task1, task2=None, *, max_docs=None, use_ocr=False,
                     ocr_engine=None, use_filters=True,
                     image_extensions=(".jpg", ".jpeg", ".png")):
    """
    Builds the FAISS index from a folder of SROIE-style documents.

    Walks `task1` recursively (subfolders included) for images, then for
    each image pairs it with :

    - An annotation file : a .txt anywhere in the tree whose content is
      ICDAR-formatted (coordinates + transcript) and whose stem matches
      the image.
    - An entity file (optional) : a .txt anywhere in the tree whose
      content is JSON and whose stem matches the image.

    Classification is done by **content sniffing**, not by folder
    layout — the SROIE corpus mixes both kinds of .txt files in
    several subfolders.

    Files lacking an annotation are either OCR'd (`use_ocr=True`) or
    silently skipped. The OCR engine is docTR by default (see
    Settings.ocr_engine) ; pass `ocr_engine="tesseract"` to switch. We
    never trigger the canonical PaddleOCR path because it's broken
    (see ingestion.py:82-99).

    Args:
        task1 (str | Path): Root folder, scanned recursively.
        task2 (str | Path, optional): Explicit folder of entity JSONs.
            Every .txt inside is forced into the entity bucket.
        max_docs (int, optional): Cap on number of documents indexed.
        use_ocr (bool): Run OCR on images without annotation.
            Otherwise they are skipped.
        ocr_engine (str, optional): OCR engine to use ("doctr" or
            "tesseract"). None uses Settings.ocr_engine (default "doctr").
        image_extensions (tuple): File extensions recognised as images.

    Returns:
        dict: Statistics about the built index.
    """

    import json

    from tqdm import tqdm

    settings = load_settings()
    ocr_engine_name = ocr_engine or getattr(settings, "ocr_engine", "doctr")
    task1 = Path(task1)
    task2 = Path(task2) if task2 else None

    images = sorted(
        f for f in task1.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    )
    if max_docs:
        images = images[:max_docs]

    # Classify EVERY .txt file by content sniff : ICDAR annotation
    # (starts with a digit) vs JSON entity (starts with `{`).  This
    # avoids the SROIE pitfall where a JSON entity .txt sits next to
    # a .jpg in a test folder and would otherwise be misread as an
    # ICDAR annotation by the canonical parser.
    annotation_index, entity_index = _classify_txt_files(task1, task2)

    print(f"\n── Ingestion : {len(images)} document(s), "
          f"{len(annotation_index)} annotation file(s), "
          f"{len(entity_index)} entity file(s) ──")

    all_passages = []
    n_skipped = n_ocr = n_annot = 0

    for img_path in tqdm(images, desc="OCR + passages"):
        stem = img_path.stem
        ann_path = annotation_index.get(stem)
        ent_path = entity_index.get(stem)

        # Extract lines : annotation (preferred) or Tesseract fallback.
        if ann_path is not None:
            lines = _ing.OCREngine.from_sroie_annotation(ann_path)
            n_annot += 1
        elif use_ocr:
            lines = _ocr.ocr_image(_ing.load_image(img_path),
                                   engine=ocr_engine, preprocess=use_filters)
            n_ocr += 1
        else:
            n_skipped += 1
            continue

        # Optional SROIE entities.
        entities = None
        if ent_path:
            try:
                with open(ent_path, encoding="utf-8", errors="ignore") as f:
                    entities = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        passages = _ing.group_lines_into_passages(
            lines=lines, source_file=str(img_path), entities=entities,
        )
        all_passages.extend(passages)

    print(f"  - {n_annot} from annotations, "
          f"{n_ocr} via OCR ({ocr_engine_name}), {n_skipped} skipped.")

    if not all_passages:
        return {
            "total_passages": 0,
            "total_documents": 0,
            "from_annotations": n_annot,
            "from_ocr": n_ocr,
            "skipped": n_skipped,
        }

    # Embeddings + FAISS via the canonical helpers (these work).
    print(f"\n── Embeddings : {len(all_passages)} passages ──")
    embedder = _ing.EmbeddingModel()
    vectors = embedder.encode(
        [p.text for p in all_passages], show_progress=True,
    )

    print(f"\n── Indexation FAISS ──")
    index = _ing.FAISSIndex()
    index.add(vectors, all_passages)
    index.save(settings.phase1_index_dir)

    docs = len({p.source_file for p in all_passages})
    print(f"\n✓ Ingestion done : {len(all_passages)} passages, {docs} documents.")
    return {
        "total_passages": len(all_passages),
        "total_documents": docs,
        "from_annotations": n_annot,
        "from_ocr": n_ocr,
        "skipped": n_skipped,
    }


### Single-document inference ###

def ask_document(file, question, *, annotation=None, top_k=5,
                 use_filters=True, embedder=None, llm_fn=None):
    """
    Answers a question about ONE document, fully in memory.

    Pipeline : load → OCR (or annotation) → chunk → MPNet embed →
    cosine top-k → Mistral via Ollama.

    Args:
        file (str | Path): Path to the source PDF or image.
        question (str): The user question.
        annotation (str | Path, optional): SROIE annotation .txt.
            When supplied, OCR is skipped.
        top_k (int): Number of passages kept for the prompt.
        use_filters (bool): Apply OCR image preprocessing (Tesseract only).
        embedder: Pre-loaded ingestion.EmbeddingModel (cache friendly).
        llm_fn: Callable (prompt) → str, injectable for tests.

    Returns:
        Answer: Text answer plus the cited Source list.
    """

    file = Path(file)
    settings = load_settings()

    lines = _extract_lines(file, annotation, use_filters=use_filters)
    if not lines:
        return Answer(text=_REFUSAL, sources=[])

    passages = _ing.group_lines_into_passages(
        lines=lines, source_file=str(file),
    )
    if not passages:
        return Answer(text=_REFUSAL, sources=[])

    embedder = embedder or _ing.EmbeddingModel()
    passage_vecs = embedder.encode([p.text for p in passages])
    query_vec = embedder.encode([question])[0]

    # MPNet vectors are L2-normalized → dot product == cosine sim.
    scores = passage_vecs @ query_vec
    order = np.argsort(-scores)[:top_k]

    # Normalise to the same dict shape as FAISSIndex.search(), so the
    # rest of the function works the same as ask_database().
    hits = [_passage_to_hit(passages[i], float(scores[i])) for i in order]

    return _answer_from_hits(question, hits, settings, llm_fn)


### Database inference ###

def ask_database(question, *, top_k=5, embedder=None, llm_fn=None,
                 index_dir=None):
    """
    Answers a question against the full FAISS index.

    Args:
        question (str): The user question.
        top_k (int): Number of passages to retrieve.
        embedder: Pre-loaded ingestion.EmbeddingModel (cache friendly).
        llm_fn: Callable (prompt) → str, injectable for tests.
        index_dir (Path, optional): Override the default index directory.

    Returns:
        Answer: Text answer plus the cited Source list.
    """

    settings = load_settings()
    index_dir = Path(index_dir) if index_dir else settings.phase1_index_dir
    if not (Path(index_dir) / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index at {index_dir}. "
            "Run `python scripts/phase1.py ingest` first."
        )

    embedder = embedder or _ing.EmbeddingModel()
    query_vec = embedder.encode([question])[0]

    # The canonical FAISSIndex already does load + search and returns
    # a list of dicts with all the fields we need (incl. "score").
    faiss_index = _ing.FAISSIndex.load(index_dir)
    hits = faiss_index.search(query_vec, k=top_k)
    if not hits:
        return Answer(text=_REFUSAL, sources=[])

    return _answer_from_hits(question, hits, settings, llm_fn)


### Helpers ###

def _classify_txt_files(task1, task2):
    """
    Scans every .txt under the given roots and classifies them by content.

    A SROIE ICDAR annotation starts with a coordinate digit (e.g.
    `10,20,100,20,...`). A SROIE entity file is JSON (starts with `{`).
    Content sniffing is far more reliable than guessing from the folder
    layout — the SROIE corpus has subfolders mixing both kinds.

    If `task2` is provided, every .txt under it is forced into the
    entity bucket regardless of content (the user told us explicitly).

    Args:
        task1 (Path): Primary root, scanned recursively.
        task2 (Path | None): Optional explicit entity folder.

    Returns:
        tuple: (annotations, entities) — two dicts keyed by stem.
    """

    annotations = {}
    entities = {}

    for txt in task1.rglob("*.txt"):
        if _txt_looks_like_json(txt):
            entities[txt.stem] = txt
        else:
            annotations[txt.stem] = txt

    if task2 is not None:
        for txt in task2.rglob("*.txt"):
            entities[txt.stem] = txt

    return annotations, entities


def _txt_looks_like_json(path):
    """
    Returns True if the file's first non-whitespace character is `{`.

    Args:
        path (Path): The .txt file to sniff.

    Returns:
        bool: True for JSON-like content, False for ICDAR-like.
    """

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.lstrip()
                if not line:
                    continue
                return line.startswith("{") or line.startswith("[")
    except OSError:
        pass
    return False


def _answer_from_hits(question, hits, settings, llm_fn):
    """
    Calls the LLM with a prompt built from retrieval hits.

    Args:
        question (str): The user question.
        hits (list[dict]): List of hit dicts, each with keys
            "text", "source_file", "page_number", "score" and
            optionally "entities".
        settings: Project settings.
        llm_fn (callable | None): Injected callable for tests.

    Returns:
        Answer: Text answer plus the cited Source list.
    """

    hits = _dedup_hits(hits)
    if not hits:
        return Answer(text=_REFUSAL, sources=[])

    prompt = _build_prompt(question, hits)
    answer_text = _call_llm(prompt, settings, llm_fn)
    sources = [_make_source(h) for h in hits]
    return Answer(text=answer_text, sources=sources)


def _dedup_hits(hits):
    """
    Removes duplicate hits keyed by (document basename, page, text).

    SROIE-style corpora often ship the same image in both train and
    test folders. Keeping all near-duplicates inflates the prompt and
    leaves the LLM citing `[1] [2]` for what is the same passage,
    confusing the answer. Deduping by basename + page + text catches
    that case without losing legitimate variants from different docs.

    Args:
        hits (list[dict]): Retrieval hits. The first occurrence of
            each (basename, page, text) triple is kept ; later ones
            are dropped.

    Returns:
        list[dict]: Deduplicated hits, original order preserved.
    """

    seen = set()
    out = []
    for h in hits:
        key = (Path(h["source_file"]).name, h["page_number"], h["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _passage_to_hit(passage, score):
    """
    Converts a canonical Passage object to the FAISSIndex.search dict shape.

    Args:
        passage (ingestion.Passage): The passage to convert.
        score (float): The similarity score.

    Returns:
        dict: A hit dict shaped like FAISSIndex.search() output.
    """

    return {
        "passage_id": passage.passage_id,
        "text": passage.text,
        "source_file": passage.source_file,
        "page_number": passage.page_number,
        "avg_confidence": passage.avg_confidence,
        "entities": passage.entities,
        "score": score,
    }


def _extract_lines(file, annotation, *, use_filters=True):
    """
    Extracts text lines from a document, via annotation or OCR.

    Args:
        file (Path): The document.
        annotation (str | Path | None): Optional SROIE annotation.
        use_filters (bool): Apply OCR image preprocessing (Tesseract only).

    Returns:
        list: ingestion.TextLine instances.
    """

    if annotation and Path(annotation).exists():
        return _ing.OCREngine.from_sroie_annotation(Path(annotation))

    if str(file).lower().endswith(".pdf"):
        images = _ing.rasterize_pdf(file)
    else:
        images = [_ing.load_image(file)]

    lines = []
    for img in images:
        lines.extend(_ocr.ocr_image(img, preprocess=use_filters))
    return lines


def _build_prompt(question, hits):
    """
    Builds the Mistral prompt with numbered passages and citation rules.

    Args:
        question (str): The user question.
        hits (list[dict]): Retrieval hits with "text", "source_file"
            and "page_number" keys.

    Returns:
        str: A complete prompt string ready for Ollama.
    """

    context_blocks = []
    for i, h in enumerate(hits, start=1):
        doc = Path(h["source_file"]).name
        context_blocks.append(
            f"[{i}] Document: « {doc} » (page {h['page_number']})\n{h['text']}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(no passages)"

    return (
        "[INST] You are a careful document assistant. Answer the question "
        "using ONLY the passages below. Cite passages with their bracketed "
        f"number, e.g. [1]. If the answer is not in the passages, reply "
        f"exactly :  {_REFUSAL}\n\n"
        f"Passages :\n{context}\n\n"
        f"Question : {question}\n[/INST]"
    )


def _call_llm(prompt, settings, llm_fn):
    """
    Runs the LLM call, defaulting to Ollama text generation.

    Args:
        prompt (str): The fully built prompt.
        settings: Project settings.
        llm_fn (callable | None): Injected callable for tests.

    Returns:
        str: The model's answer.
    """

    if llm_fn is not None:
        return llm_fn(prompt)
    return _llm.generate_text(
        prompt,
        model=settings.ollama_text_model,
        host=settings.ollama_host,
        keep_alive=settings.ollama_keep_alive,
    )


def _make_source(hit, *, max_chars=400):
    """
    Converts a retrieval hit dict into a Source for display.

    Args:
        hit (dict): A hit dict with "text", "source_file", "page_number",
            "score" and optional "entities" keys.
        max_chars (int): Max characters of the snippet.

    Returns:
        Source: Ready-to-render Source instance.
    """

    text = hit["text"]
    snippet = text if len(text) <= max_chars else text[: max_chars - 1] + "…"
    return Source(
        document=Path(hit["source_file"]).name,
        page=hit["page_number"],
        score=hit["score"],
        snippet=snippet,
        entities=hit.get("entities"),
    )
