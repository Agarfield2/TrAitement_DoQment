"""
Tests for doqment.phase1.ask_document and ask_database.

We mock the OCR (so we don't need tesseract) and the LLM (so we don't
need Ollama). The MPNet embedder is replaced by a deterministic fake
so HuggingFace is never contacted.
"""

import numpy as np
import pytest
from PIL import Image


### Fakes ###

def _fake_lines():
    from ingestion import BoundingBox, TextLine

    return [
        TextLine("ACME STORE",
                 BoundingBox(10, 10, 100, 10, 100, 30, 10, 30), 0.95),
        TextLine("TOTAL : 42.50",
                 BoundingBox(10, 50, 200, 50, 200, 70, 10, 70), 0.91),
    ]


class _FakeEmbedder:
    """Deterministic unit-vector embedder."""

    def encode(self, texts, **kwargs):
        rng = np.random.default_rng(seed=42)
        arr = rng.standard_normal((len(texts), 8)).astype("float32")
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def _fake_llm(prompt):
    return "The total is 42.50 [1]"


### ask_document ###

def test_ask_document_returns_answer_and_sources(tmp_path, monkeypatch):
    from doqment import ocr, phase1

    file_path = tmp_path / "receipt.png"
    Image.new("RGB", (50, 50), "white").save(file_path)

    monkeypatch.setattr(ocr, "ocr_image", lambda img, **kw: _fake_lines())

    answer = phase1.ask_document(
        file=file_path, question="What is the total?",
        embedder=_FakeEmbedder(), llm_fn=_fake_llm,
    )
    assert answer.text == "The total is 42.50 [1]"
    assert len(answer.sources) >= 1
    s = answer.sources[0]
    assert s.document == "receipt.png"
    assert s.page == 1
    assert isinstance(s.score, float)
    assert isinstance(s.snippet, str) and s.snippet


def test_ask_document_refuses_when_ocr_empty(tmp_path, monkeypatch):
    from doqment import ocr, phase1

    file_path = tmp_path / "blank.png"
    Image.new("RGB", (50, 50), "white").save(file_path)

    monkeypatch.setattr(ocr, "ocr_image", lambda img, **kw: [])

    answer = phase1.ask_document(
        file=file_path, question="?",
        embedder=_FakeEmbedder(), llm_fn=_fake_llm,
    )
    assert answer.sources == []
    assert "do not have this information" in answer.text.lower()


def test_ask_document_uses_annotation_when_present(tmp_path, monkeypatch):
    """When a valid annotation is supplied, OCR must be skipped."""

    from doqment import ocr, phase1

    file_path = tmp_path / "receipt.png"
    Image.new("RGB", (50, 50), "white").save(file_path)

    annotation_path = tmp_path / "receipt.txt"
    annotation_path.write_text(
        "10,10,100,10,100,30,10,30,ACME STORE\n"
        "10,50,200,50,200,70,10,70,TOTAL : 42.50\n",
        encoding="utf-8",
    )

    ocr_called = {"hit": False}

    def _bad_ocr(img, **kw):
        ocr_called["hit"] = True
        return []

    monkeypatch.setattr(ocr, "ocr_image", _bad_ocr)

    answer = phase1.ask_document(
        file=file_path, question="?",
        annotation=annotation_path,
        embedder=_FakeEmbedder(), llm_fn=_fake_llm,
    )
    assert ocr_called["hit"] is False
    assert len(answer.sources) >= 1


def test_ask_document_passes_llm_fn_a_prompt_with_citations(
    tmp_path, monkeypatch,
):
    """The prompt sent to the LLM must contain the bracketed-index format."""

    from doqment import ocr, phase1

    file_path = tmp_path / "receipt.png"
    Image.new("RGB", (50, 50), "white").save(file_path)
    monkeypatch.setattr(ocr, "ocr_image", lambda img, **kw: _fake_lines())

    captured = {}

    def _capture(prompt):
        captured["prompt"] = prompt
        return "ok"

    phase1.ask_document(
        file=file_path, question="What is the total?",
        embedder=_FakeEmbedder(), llm_fn=_capture,
    )
    assert "[1]" in captured["prompt"]
    assert "What is the total?" in captured["prompt"]


### ask_database (no real index, just verify the FileNotFoundError path) ###

def test_ask_database_raises_if_index_missing(tmp_path):
    from doqment import phase1

    with pytest.raises(FileNotFoundError, match="No FAISS index"):
        phase1.ask_database(
            question="?", embedder=_FakeEmbedder(), llm_fn=_fake_llm,
            index_dir=tmp_path / "nope",
        )


def test_ask_database_returns_answer_when_index_present(tmp_path, monkeypatch):
    """End-to-end with a tiny hand-built FAISS index, using the
    canonical FAISSIndex so the on-disk format is exactly the real
    one (this would have caught the metadata.pkl shape bug)."""

    from ingestion import FAISSIndex, Passage
    from doqment import phase1

    embedder = _FakeEmbedder()
    texts = ["ACME STORE", "TOTAL : 42.50"]

    # Use the same dim as the fake embedder so FAISSIndex stays consistent.
    monkeypatch.setattr(FAISSIndex, "__init__",
                        lambda self, dim=8: _faiss_index_init(self, dim))

    vecs = embedder.encode(texts).astype("float32")
    passages = [
        Passage(
            passage_id=f"p{i}", text=t,
            source_file="receipt.jpg", page_number=1,
            bboxes=[], entities=None, avg_confidence=1.0,
        )
        for i, t in enumerate(texts)
    ]

    index = FAISSIndex()
    index.add(vecs, passages)

    index_dir = tmp_path / "processed"
    index.save(index_dir)

    answer = phase1.ask_database(
        question="total", top_k=2,
        embedder=embedder, llm_fn=_fake_llm,
        index_dir=index_dir,
    )
    assert answer.text == "The total is 42.50 [1]"
    assert len(answer.sources) == 2
    # The canonical FAISSIndex.search returns dicts ; our wrapper must
    # convert them to Source dataclasses with the right field names.
    s = answer.sources[0]
    assert s.document == "receipt.jpg"
    assert s.page == 1
    assert isinstance(s.score, float)
    assert isinstance(s.snippet, str) and s.snippet


def _faiss_index_init(self, dim):
    """Init helper that mirrors FAISSIndex.__init__ with a smaller dim."""

    self.dim = dim
    self._index = None
    self._metadata = []


### Tests : ingest_directory (no real LLM / no real embedder load) ###

def test_ingest_directory_uses_annotations_when_present(tmp_path, monkeypatch):
    """
    With a SROIE-style annotation next to the image, ingest_directory
    must use the canonical annotation parser — never trigger Tesseract
    nor the broken canonical OCREngine.
    """

    import doqment.phase1 as phase1_mod
    from PIL import Image

    task1 = tmp_path / "task1"
    task1.mkdir()
    img = task1 / "receipt.jpg"
    Image.new("RGB", (50, 50), "white").save(img)
    (task1 / "receipt.txt").write_text(
        "10,10,100,10,100,30,10,30,ACME STORE\n"
        "10,50,200,50,200,70,10,70,TOTAL : 42.50\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(phase1_mod, "load_settings",
                        lambda: type("S", (), {"phase1_index_dir": tmp_path / "out"})())

    # The Tesseract wrapper must NEVER be called when an annotation exists.
    from doqment import ocr
    ocr_called = {"hit": False}

    def _trap(img, **kw):
        ocr_called["hit"] = True
        return []

    monkeypatch.setattr(ocr, "ocr_image", _trap)

    # Replace the real MPNet embedder and FAISS to avoid HuggingFace I/O.
    monkeypatch.setattr(
        phase1_mod._ing, "EmbeddingModel",
        lambda *a, **kw: _FakeEmbedderWithProgress(),
    )
    monkeypatch.setattr(
        phase1_mod._ing, "FAISSIndex", _FakeFAISS,
    )

    stats = phase1_mod.ingest_directory(task1, max_docs=1)

    assert ocr_called["hit"] is False
    assert stats["total_passages"] >= 1
    assert stats["from_annotations"] == 1
    assert stats["from_tesseract"] == 0
    assert stats["skipped"] == 0


def test_ingest_directory_skips_unannotated_by_default(tmp_path, monkeypatch):
    """
    Without --tesseract, files lacking an annotation must be silently
    skipped (the broken canonical PaddleOCR path is never engaged).
    """

    import doqment.phase1 as phase1_mod
    from PIL import Image

    task1 = tmp_path / "task1"
    task1.mkdir()
    img = task1 / "receipt.jpg"
    Image.new("RGB", (50, 50), "white").save(img)
    # NO annotation .txt next to it.

    monkeypatch.setattr(phase1_mod, "load_settings",
                        lambda: type("S", (), {"phase1_index_dir": tmp_path / "out"})())

    from doqment import ocr
    ocr_called = {"hit": False}

    def _trap(img, **kw):
        ocr_called["hit"] = True
        return []

    monkeypatch.setattr(ocr, "ocr_image", _trap)

    monkeypatch.setattr(
        phase1_mod._ing, "EmbeddingModel",
        lambda *a, **kw: _FakeEmbedderWithProgress(),
    )
    monkeypatch.setattr(
        phase1_mod._ing, "FAISSIndex", _FakeFAISS,
    )

    stats = phase1_mod.ingest_directory(task1)

    assert ocr_called["hit"] is False
    assert stats["skipped"] == 1
    assert stats["total_passages"] == 0


def test_ingest_directory_uses_tesseract_when_requested(tmp_path, monkeypatch):
    """
    With --tesseract on, unannotated files go through our Tesseract
    wrapper (never the broken canonical OCREngine).
    """

    import doqment.phase1 as phase1_mod
    from PIL import Image

    task1 = tmp_path / "task1"
    task1.mkdir()
    img = task1 / "receipt.jpg"
    Image.new("RGB", (50, 50), "white").save(img)

    monkeypatch.setattr(phase1_mod, "load_settings",
                        lambda: type("S", (), {"phase1_index_dir": tmp_path / "out"})())

    from doqment import ocr
    monkeypatch.setattr(ocr, "ocr_image", lambda img, **kw: _fake_lines())

    monkeypatch.setattr(
        phase1_mod._ing, "EmbeddingModel",
        lambda *a, **kw: _FakeEmbedderWithProgress(),
    )
    monkeypatch.setattr(
        phase1_mod._ing, "FAISSIndex", _FakeFAISS,
    )

    stats = phase1_mod.ingest_directory(task1, use_tesseract=True)

    assert stats["from_tesseract"] == 1
    assert stats["skipped"] == 0
    assert stats["total_passages"] >= 1


### Test helpers for ingest_directory tests ###

class _FakeEmbedderWithProgress:
    """Embedder with a `show_progress` keyword the canonical uses."""

    def encode(self, texts, **kwargs):
        return _FakeEmbedder().encode(texts)


class _FakeFAISS:
    """In-memory replacement for ingestion.FAISSIndex used by tests."""

    def add(self, vectors, passages):
        self.vectors, self.passages = vectors, list(passages)

    def save(self, path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)


### Tests : recursive ingestion + entity auto-discovery ###

def test_ingest_walks_subdirectories(tmp_path, monkeypatch):
    """
    Pointing at a parent folder must find images in subfolders too.
    """

    import doqment.phase1 as phase1_mod
    from PIL import Image

    root = tmp_path / "data" / "SROIE2019"
    sub = root / "task1train"
    sub.mkdir(parents=True)
    img = sub / "receipt.jpg"
    Image.new("RGB", (50, 50), "white").save(img)
    (sub / "receipt.txt").write_text(
        "10,10,100,10,100,30,10,30,ACME\n", encoding="utf-8",
    )

    monkeypatch.setattr(phase1_mod, "load_settings",
                        lambda: type("S", (), {"phase1_index_dir": tmp_path / "out"})())
    monkeypatch.setattr(
        phase1_mod._ing, "EmbeddingModel",
        lambda *a, **kw: _FakeEmbedderWithProgress(),
    )
    monkeypatch.setattr(phase1_mod._ing, "FAISSIndex", _FakeFAISS)

    # Point at the GRANDPARENT, not the subfolder with images.
    stats = phase1_mod.ingest_directory(root)

    assert stats["from_annotations"] == 1
    assert stats["total_documents"] == 1


def test_ingest_auto_discovers_entity_files(tmp_path, monkeypatch):
    """
    Without an explicit task2, entity .txt files (no image sibling)
    discovered anywhere in the tree must be picked up.
    """

    import doqment.phase1 as phase1_mod
    from PIL import Image

    root = tmp_path / "SROIE2019"
    img_dir = root / "task1train"
    ent_dir = root / "task2train"
    img_dir.mkdir(parents=True)
    ent_dir.mkdir(parents=True)

    img = img_dir / "X51005.jpg"
    Image.new("RGB", (50, 50), "white").save(img)
    (img_dir / "X51005.txt").write_text(
        "10,10,100,10,100,30,10,30,ACME\n", encoding="utf-8",
    )
    (ent_dir / "X51005.txt").write_text(
        '{"company": "ACME", "total": "42.50"}', encoding="utf-8",
    )

    monkeypatch.setattr(phase1_mod, "load_settings",
                        lambda: type("S", (), {"phase1_index_dir": tmp_path / "out"})())
    monkeypatch.setattr(
        phase1_mod._ing, "EmbeddingModel",
        lambda *a, **kw: _FakeEmbedderWithProgress(),
    )
    monkeypatch.setattr(phase1_mod._ing, "FAISSIndex", _FakeFAISS)

    stats = phase1_mod.ingest_directory(root)

    assert stats["from_annotations"] == 1


def test_classify_distinguishes_icdar_from_json(tmp_path):
    """
    ICDAR annotation .txt and JSON entity .txt must end up in
    different buckets, even when they share a stem AND sit next to
    each other (a case the previous heuristic was getting wrong).
    """

    from doqment.phase1 import _classify_txt_files

    root = tmp_path
    img_dir = root / "task1"
    ent_dir = root / "task2"
    img_dir.mkdir()
    ent_dir.mkdir()

    # ICDAR annotation : starts with a digit-coord, lives next to image.
    (img_dir / "doc.txt").write_text(
        "10,20,100,20,100,40,10,40,ACME STORE\n",
        encoding="utf-8",
    )
    # JSON entity : starts with `{`, lives in a sibling folder.
    (ent_dir / "doc.txt").write_text(
        '{"company": "ACME", "total": "42.50"}',
        encoding="utf-8",
    )

    annotations, entities = _classify_txt_files(root, task2=None)

    assert annotations["doc"] == img_dir / "doc.txt"
    assert entities["doc"] == ent_dir / "doc.txt"


def test_classify_routes_inline_json_next_to_image_to_entities(tmp_path):
    """
    Real bug we hit on SROIE : a JSON .txt sat right next to a .jpg in
    a test subfolder. With the OLD heuristic ("same folder as image →
    annotation") this crashed the ICDAR parser. Content sniffing must
    route it to the entity bucket regardless of location.
    """

    from doqment.phase1 import _classify_txt_files

    root = tmp_path / "mixed"
    root.mkdir()
    (root / "doc.jpg").write_bytes(b"")
    (root / "doc.txt").write_text(
        '{"address": "S117, JLN BESAR"}',
        encoding="utf-8",
    )

    annotations, entities = _classify_txt_files(root, task2=None)

    assert "doc" not in annotations
    assert entities["doc"] == root / "doc.txt"


def test_classify_explicit_task2_forces_entity_bucket(tmp_path):
    """
    Anything passed via explicit task2 must land in the entity bucket
    regardless of content (the user told us so).
    """

    from doqment.phase1 import _classify_txt_files

    task1 = tmp_path / "task1"
    task2 = tmp_path / "task2"
    task1.mkdir()
    task2.mkdir()
    # An ICDAR-looking file in task2 — still treated as entity because
    # the caller chose to mark task2 as such.
    (task2 / "doc.txt").write_text(
        "10,20,100,20,100,40,10,40,SHOULD-NOT-CRASH-ICDAR\n",
        encoding="utf-8",
    )

    annotations, entities = _classify_txt_files(task1, task2=task2)
    assert "doc" in entities
    assert "doc" not in annotations


def test_txt_looks_like_json_skips_whitespace_and_bom():
    """
    The sniffer must be tolerant of leading whitespace, blank lines
    and UTF-8 BOM markers.
    """

    import tempfile
    from doqment.phase1 import _txt_looks_like_json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        json_with_bom = Path(tmp) / "json.txt"
        json_with_bom.write_text("\n\n  {\"x\": 1}", encoding="utf-8")
        assert _txt_looks_like_json(json_with_bom) is True

        icdar = Path(tmp) / "icdar.txt"
        icdar.write_text("\n10,20,100,20,100,40,10,40,HELLO\n", encoding="utf-8")
        assert _txt_looks_like_json(icdar) is False

        empty = Path(tmp) / "empty.txt"
        empty.write_text("", encoding="utf-8")
        assert _txt_looks_like_json(empty) is False


### Tests : source deduplication ###

def test_dedup_collapses_train_test_duplicates():
    """
    SROIE ships the same receipt image in both train and test folders.
    FAISS retrieves both copies with identical text. Dedup must keep
    only the first one so the LLM doesn't cite [1][2] for the same
    passage.
    """

    from doqment.phase1 import _dedup_hits

    hits = [
        {
            "source_file": "data/SROIE2019/0325updated.task1train(626p)/X51006913023.jpg",
            "page_number": 1,
            "text": "TOTAL AMOUNT: $8.20",
            "score": 0.577,
        },
        {
            "source_file": "data/SROIE2019/task1&2_test(361p)/X51006913023.jpg",
            "page_number": 1,
            "text": "TOTAL AMOUNT: $8.20",
            "score": 0.577,
        },
        {
            "source_file": "data/SROIE2019/0325updated.task1train(626p)/X51007846304.jpg",
            "page_number": 1,
            "text": "TOTAL AMOUNT: RM7.52",
            "score": 0.607,
        },
    ]

    out = _dedup_hits(hits)
    assert len(out) == 2
    assert out[0]["text"] == "TOTAL AMOUNT: $8.20"
    assert out[1]["text"] == "TOTAL AMOUNT: RM7.52"


def test_dedup_keeps_different_pages_same_doc():
    """Same doc, same text, different page → keep both."""

    from doqment.phase1 import _dedup_hits

    hits = [
        {"source_file": "report.pdf", "page_number": 1, "text": "Summary"},
        {"source_file": "report.pdf", "page_number": 5, "text": "Summary"},
    ]
    assert len(_dedup_hits(hits)) == 2


def test_dedup_keeps_different_text_same_doc_same_page():
    """Same doc, same page, different text → keep both."""

    from doqment.phase1 import _dedup_hits

    hits = [
        {"source_file": "doc.jpg", "page_number": 1, "text": "first passage"},
        {"source_file": "doc.jpg", "page_number": 1, "text": "second passage"},
    ]
    assert len(_dedup_hits(hits)) == 2
