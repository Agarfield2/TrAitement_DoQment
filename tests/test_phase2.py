"""
Tests for doqment.phase2.

ColQwen2 weights are ~3 GB and Qwen2.5-VL needs Ollama running, so we
do not exercise them. We test :

- file discovery and md5 hashing
- the metadata store (idempotence contract)
- the rasterization helpers on synthetic input
- the maxsim scorer
"""

import sqlite3

import pytest


### File discovery ###

def test_iter_supported_files_finds_pdf_and_images(tmp_path):
    from doqment import phase2

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "c.jpg").write_bytes(b"")
    (tmp_path / "ignored.txt").write_text("nope")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.jpeg").write_bytes(b"")

    found = sorted(p.name for p in phase2._iter_supported_files(tmp_path))
    assert found == ["a.pdf", "b.png", "c.jpg", "d.jpeg"]


### file_md5 ###

def test_file_md5_is_stable(tmp_path):
    from doqment.phase2_store import file_md5

    f = tmp_path / "x.bin"
    f.write_bytes(b"the same bytes")
    md5_a = file_md5(f)
    md5_b = file_md5(f)
    assert md5_a == md5_b
    assert len(md5_a) == 32   # hex digest


def test_file_md5_differs_for_different_content(tmp_path):
    from doqment.phase2_store import file_md5

    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"AAA")
    b.write_bytes(b"BBB")
    assert file_md5(a) != file_md5(b)


### MetadataStore ###

def test_metadata_store_idempotence(tmp_path):
    from doqment.phase2_store import MetadataStore

    db = tmp_path / "meta.db"
    store = MetadataStore(db)

    assert store.already_ingested("abcdef") is False
    store.add_document("doc1", tmp_path / "doc1.pdf", "abcdef", 3, "pdf_native")
    assert store.already_ingested("abcdef") is True
    assert store.already_ingested("xyz") is False
    store.close()


def test_metadata_store_persists_across_reopen(tmp_path):
    from doqment.phase2_store import MetadataStore

    db = tmp_path / "meta.db"
    store = MetadataStore(db)
    store.add_document("doc1", tmp_path / "doc1.pdf", "abcdef", 3, "pdf_native")
    store.close()

    store2 = MetadataStore(db)
    assert store2.already_ingested("abcdef") is True
    store2.close()


def test_metadata_store_persists_page_rows(tmp_path):
    from doqment.phase2_store import MetadataStore

    db = tmp_path / "meta.db"
    store = MetadataStore(db)
    store.add_document("doc1", tmp_path / "doc1.pdf", "abc", 2, "pdf_native")
    store.add_page("page1", "doc1", 1, tmp_path / "p1.png", 200)
    store.add_page("page2", "doc1", 2, tmp_path / "p2.png", 200)
    store.close()

    with sqlite3.connect(str(db)) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM pages WHERE doc_id = ?", ("doc1",),
        ).fetchone()[0]
    assert count == 2


### detect_source_kind ###

def test_detect_source_kind_for_image(tmp_path):
    from doqment.phase2_store import SourceKind, detect_source_kind
    from PIL import Image

    p = tmp_path / "x.png"
    Image.new("RGB", (10, 10)).save(p)
    assert detect_source_kind(p) == SourceKind.IMAGE


def test_detect_source_kind_unsupported_raises(tmp_path):
    from doqment.phase2_store import detect_source_kind

    p = tmp_path / "x.docx"
    p.write_bytes(b"not a real docx")
    with pytest.raises(ValueError, match="Unsupported file format"):
        detect_source_kind(p)


### normalize_image ###

def test_normalize_image_produces_rgb_png(tmp_path):
    from doqment.phase2_store import normalize_image
    from PIL import Image

    src = tmp_path / "input.jpg"
    Image.new("L", (20, 20)).save(src)   # grayscale on purpose

    out_dir = tmp_path / "pages"
    out = normalize_image(src, out_dir)

    assert out.exists()
    assert out.suffix == ".png"
    assert Image.open(str(out)).mode == "RGB"


### maxsim ###

def test_maxsim_basic_score():
    import torch
    from doqment.phase2_store import maxsim

    # Query : 2 tokens, dim 4.  Page : 3 patches, dim 4.
    query = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    page = torch.tensor([
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    # max per query : 0.5 then 0.8, sum 1.3.
    assert maxsim(query, page) == pytest.approx(1.3, abs=1e-5)


def test_maxsim_zero_when_orthogonal():
    import torch
    from doqment.phase2_store import maxsim

    query = torch.tensor([[1.0, 0.0]])
    page = torch.tensor([[0.0, 1.0], [0.0, 0.5]])
    assert maxsim(query, page) == pytest.approx(0.0)


### _page_num parsing ###

def test_page_num_parses_suffix():
    from pathlib import Path
    from doqment.phase2 import _page_num

    assert _page_num(Path("/tmp/foo_p001.png")) == 1
    assert _page_num(Path("/tmp/foo_p042.png")) == 42


def test_page_num_returns_one_on_unrecognized_name():
    from pathlib import Path
    from doqment.phase2 import _page_num

    assert _page_num(Path("/tmp/random.png")) == 1


### Ingestion : file-not-found ###

def test_ingest_directory_raises_if_dir_missing(tmp_path):
    from doqment import phase2

    with pytest.raises(FileNotFoundError):
        phase2.ingest_directory(tmp_path / "nope")
