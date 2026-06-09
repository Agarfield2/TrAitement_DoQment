"""
Pipeline 2 — multimodal RAG.

Three entry points, symmetric with Pipeline 1 :

- `ingest_directory(...)`        ingests every PDF/image into Qdrant.
- `ask_document(file, question)` answers a question about ONE document.
- `ask_database(question)`       answers using the whole Qdrant index.

The Qdrant database is local-path (no daemon required). Generation is
delegated to a vision-language model served by Ollama (Qwen2.5-VL by
default).
"""

import uuid
from dataclasses import dataclass
from pathlib import Path

from doqment import llm as _llm
from doqment import phase2_store as _store
from doqment.settings import load_settings


### Public Types ###

@dataclass(frozen=True)
class Source:
    """
    A page cited in an answer.

    Attributes:
        document: Source file basename (no path).
        page: 1-based page number.
        score: MaxSim score.
        image_path: Absolute path to the rasterized PNG of that page.
    """

    document: str
    page: int
    score: float
    image_path: Path


@dataclass(frozen=True)
class Answer:
    """
    The full result of a question.

    Attributes:
        text: The VLM's answer.
        sources: Pages retrieved and shown to the VLM (in order).
        cited: Subset of `sources` actually cited by the VLM.
    """

    text: str
    sources: list
    cited: list


_REFUSAL = "Information non trouvée dans les documents fournis."


### Ingestion ###

def ingest_directory(src_dir, *, encoder=None):
    """
    Indexes every supported file under `src_dir` into Qdrant.

    Idempotent : files whose MD5 already lives in metadata.db are skipped
    without re-encoding.

    Args:
        src_dir (str | Path): Folder of PDFs and images to ingest.
        encoder: A pre-built ColQwen2Encoder, useful for tests and for
            avoiding the 10 s warm-up between calls.

    Returns:
        int: Number of new pages indexed (0 if everything was up to date).
    """

    settings = load_settings()
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory : {src_dir}")

    encoder = encoder or _store.ColQwen2Encoder(
        model_name=settings.colqwen_model,
        device=settings.colqwen_device,
        dtype=settings.colqwen_dtype,
    )

    from tqdm import tqdm

    files = sorted(_iter_supported_files(src_dir))
    print(f"  {len(files)} fichier(s) — encodage ColQwen2 sur "
          f"'{settings.colqwen_device}' (dtype {settings.colqwen_dtype})")

    n_pages = 0
    with _store.open_stores(settings) as (vec_store, meta_store):
        for file in tqdm(files, desc="ColQwen2 encode", unit="file"):
            n_pages += _ingest_one_file(
                file, encoder, vec_store, meta_store, settings,
            )
    return n_pages


### Single-document inference ###

def ask_document(file, question, *, encoder=None, generate_k=None):
    """
    Answers a question about ONE document, fully in memory.

    Pipeline : rasterize → ColQwen2 encode → brute-force MaxSim →
    top-k page images → Qwen2.5-VL via Ollama. Nothing is persisted.

    Args:
        file (str | Path): Source PDF or image.
        question (str): The user question.
        encoder: Pre-loaded ColQwen2Encoder (cache friendly).
        generate_k (int, optional): Pages sent to the VLM (defaults to
            settings.generate_k).

    Returns:
        Answer: The VLM answer plus the cited Source list.
    """

    settings = load_settings()
    file = Path(file)
    generate_k = generate_k or settings.generate_k

    encoder = encoder or _store.ColQwen2Encoder(
        model_name=settings.colqwen_model,
        device=settings.colqwen_device,
        dtype=settings.colqwen_dtype,
    )

    tmp_dir = settings.phase2_pages_dir / f"_tmp_{uuid.uuid4().hex[:8]}"
    image_paths, _ = _rasterize_for_query(file, tmp_dir, settings)
    images = [_store.load_page_image(p) for p in image_paths]

    page_embs = encoder.encode_pages(images)
    query_emb = encoder.encode_query(question)

    scored = [
        (path, _store.maxsim(query_emb, emb))
        for path, emb in zip(image_paths, page_embs)
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:generate_k]

    sources = [
        Source(document=file.name, page=_page_num(path), score=score,
               image_path=path)
        for path, score in top
    ]
    return _generate_answer(question, sources, settings)


### Database inference ###

def ask_database(question, *, encoder=None, retrieve_k=None, generate_k=None):
    """
    Answers a question against the whole indexed Qdrant database.

    Args:
        question (str): The user question.
        encoder: Pre-loaded ColQwen2Encoder.
        retrieve_k (int, optional): Pages pulled from Qdrant.
        generate_k (int, optional): Pages actually sent to the VLM.

    Returns:
        Answer: The VLM answer plus the cited Source list.
    """

    settings = load_settings()
    retrieve_k = retrieve_k or settings.retrieve_k
    generate_k = generate_k or settings.generate_k

    encoder = encoder or _store.ColQwen2Encoder(
        model_name=settings.colqwen_model,
        device=settings.colqwen_device,
        dtype=settings.colqwen_dtype,
    )
    query_emb = encoder.encode_query(question)
    query_vecs = query_emb.tolist()

    with _store.open_stores(settings) as (vec_store, _meta_store):
        hits = vec_store.search(query_vecs, top_k=retrieve_k)

    sources = []
    for hit in hits[:generate_k]:
        payload = hit["payload"]
        sources.append(Source(
            document=payload["document"],
            page=payload["page_num"],
            score=hit["score"],
            image_path=Path(payload["image_path"]),
        ))

    return _generate_answer(question, sources, settings)


### Helpers ###

def _iter_supported_files(folder):
    """
    Yields supported document paths under a folder, recursively.

    Args:
        folder (Path): The root folder to scan.

    Yields:
        Path: One path per supported file (PDF or image).
    """

    supported = {".pdf"} | _store._IMAGE_EXTS
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in supported:
            yield p


def _ingest_one_file(file, encoder, vec_store, meta_store, settings):
    """
    Indexes one file, skipping if its MD5 is already in the database.

    Args:
        file (Path): Source file.
        encoder (ColQwen2Encoder): Pre-built encoder.
        vec_store (QdrantStore): Where vectors go.
        meta_store (MetadataStore): Where idempotence metadata goes.
        settings (Settings): Project settings.

    Returns:
        int: Number of pages newly indexed (0 if skipped).
    """

    md5 = _store.file_md5(file)
    if meta_store.already_ingested(md5):
        return 0

    doc_id = file.stem
    image_paths, kind = _rasterize_for_query(file, settings.phase2_pages_dir, settings)
    if not image_paths:
        return 0

    images = [_store.load_page_image(p) for p in image_paths]
    page_embs = encoder.encode_pages(images)

    for i, (path, emb) in enumerate(zip(image_paths, page_embs), start=1):
        page_id = uuid.uuid4()
        vec_store.upsert_page(
            page_id=page_id,
            vectors=emb.tolist(),
            payload={
                "doc_id": doc_id,
                "document": file.name,
                "page_num": i,
                "image_path": str(path.resolve()),
                "source_kind": kind.value,
            },
        )
        meta_store.add_page(page_id, doc_id, i, path, _dpi_for(kind))

    meta_store.add_document(doc_id, file, md5, len(image_paths), kind.value)
    return len(image_paths)


def _rasterize_for_query(file, output_dir, settings):
    """
    Rasterizes a single file at the appropriate DPI.

    Args:
        file (Path): The source PDF or image.
        output_dir (Path): Where to drop the PNGs.
        settings (Settings): Unused for now ; kept for symmetry.

    Returns:
        tuple: (list of PNG paths, SourceKind).
    """

    kind = _store.detect_source_kind(file)
    if kind == _store.SourceKind.IMAGE:
        return [_store.normalize_image(file, output_dir)], kind

    paths = _store.rasterize_pdf(file, output_dir, dpi=_dpi_for(kind))
    return paths, kind


def _dpi_for(kind):
    """
    Returns the rasterization DPI for a given SourceKind.

    Args:
        kind (SourceKind): The source kind.

    Returns:
        int: The DPI.
    """

    return 200 if kind == _store.SourceKind.PDF_NATIVE else 300


def _page_num(image_path):
    """
    Extracts the page number from a `<stem>_p<NNN>.png` file name.

    Args:
        image_path (Path): The rasterized page path.

    Returns:
        int: The page number (1-based), or 1 on parse failure.
    """

    stem = image_path.stem
    try:
        return int(stem.rsplit("_p", 1)[-1])
    except ValueError:
        return 1


def _generate_answer(question, sources, settings):
    """
    Calls the VLM on the sources and builds the final Answer.

    Args:
        question (str): The user question.
        sources (list[Source]): Pages selected for generation.
        settings (Settings): Project settings.

    Returns:
        Answer: Text + sources + cited subset.
    """

    if not sources:
        return Answer(text=_REFUSAL, sources=[], cited=[])

    images = [_store.load_page_image(s.image_path) for s in sources]
    text = _llm.generate_vision(
        prompt=question,
        images=images,
        model=settings.ollama_vision_model,
        host=settings.ollama_host,
        keep_alive=settings.ollama_keep_alive,
        num_ctx=settings.ollama_num_ctx,
        image_max_side=settings.vlm_image_max_side,
    ) or _REFUSAL

    # Le modèle ne s'auto-cite plus : les pages envoyées au VLM (déjà les
    # mieux classées par Qdrant) constituent les sources qui appuient la réponse.
    return Answer(text=text, sources=sources, cited=sources)
