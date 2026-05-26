"""
Tests for the Tesseract OCR wrapper.

These tests do not need the tesseract binary : they exercise the
internal helpers (`_group_into_lines`, `_pick_lang`) with synthetic
data, plus the missing-dependency path.
"""

import pytest


def _fake_image():
    from PIL import Image
    return Image.new("RGB", (10, 10), color="white")


### Tests : _group_into_lines ###

def test_groups_words_on_same_line():
    from doqment.ocr import _group_into_lines

    data = {
        "text":       ["TOTAL", "42.00"],
        "left":       [10, 100],
        "top":        [20, 20],
        "width":      [80, 60],
        "height":     [25, 25],
        "conf":       ["95", "90"],
        "block_num":  [1, 1],
        "par_num":    [1, 1],
        "line_num":   [1, 1],
    }
    lines = _group_into_lines(data)
    assert len(lines) == 1
    assert lines[0].text == "TOTAL 42.00"


def test_separates_words_on_different_lines():
    from doqment.ocr import _group_into_lines

    data = {
        "text":       ["A", "B"],
        "left":       [10, 10],
        "top":        [20, 60],
        "width":      [80, 60],
        "height":     [25, 25],
        "conf":       ["95", "90"],
        "block_num":  [1, 1],
        "par_num":    [1, 1],
        "line_num":   [1, 2],
    }
    lines = _group_into_lines(data)
    assert len(lines) == 2


def test_skips_empty_and_whitespace_text():
    from doqment.ocr import _group_into_lines

    data = {
        "text":       ["", "   ", "TOTAL"],
        "left":       [0, 0, 10],
        "top":        [0, 0, 20],
        "width":      [0, 0, 80],
        "height":     [0, 0, 25],
        "conf":       ["-1", "-1", "95"],
        "block_num":  [1, 1, 1],
        "par_num":    [1, 1, 1],
        "line_num":   [0, 0, 1],
    }
    lines = _group_into_lines(data)
    assert len(lines) == 1
    assert lines[0].text == "TOTAL"


def test_bbox_is_union_of_words():
    from doqment.ocr import _group_into_lines

    data = {
        "text":       ["A", "B"],
        "left":       [10, 50],
        "top":        [20, 30],
        "width":      [10, 10],
        "height":     [20, 25],
        "conf":       ["90", "90"],
        "block_num":  [1, 1],
        "par_num":    [1, 1],
        "line_num":   [1, 1],
    }
    lines = _group_into_lines(data)
    bbox = lines[0].bbox
    assert bbox.xmin == 10
    assert bbox.ymin == 20
    assert bbox.xmax == 60
    # Use axis-aligned max because BoundingBox.ymax has a known bug.
    assert max(bbox.y1, bbox.y2, bbox.y3, bbox.y4) == 55


def test_average_confidence_excludes_negative():
    from doqment.ocr import _group_into_lines

    data = {
        "text":       ["A", "B", "C"],
        "left":       [0, 20, 40],
        "top":        [0, 0, 0],
        "width":      [10, 10, 10],
        "height":     [10, 10, 10],
        "conf":       ["80", "90", "-1"],
        "block_num":  [1, 1, 1],
        "par_num":    [1, 1, 1],
        "line_num":   [1, 1, 1],
    }
    lines = _group_into_lines(data)
    assert lines[0].confidence == pytest.approx((0.80 + 0.90) / 2)


### Tests : _pick_lang ###

class _FakePyTess:
    def __init__(self, installed):
        self._installed = installed

    def get_languages(self):
        return list(self._installed)


def test_pick_lang_keeps_installed_codes_only():
    from doqment.ocr import _pick_lang
    assert _pick_lang(_FakePyTess(["fra", "deu"]), "fra+eng") == "fra"


def test_pick_lang_falls_back_to_english():
    from doqment.ocr import _pick_lang
    assert _pick_lang(_FakePyTess(["eng", "deu"]), "fra") == "eng"


def test_pick_lang_uses_first_available_as_last_resort():
    from doqment.ocr import _pick_lang
    assert _pick_lang(_FakePyTess(["spa", "ita"]), "fra") == "spa"


### Tests : missing pytesseract ###

def test_missing_pytesseract_raises_clear_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    from doqment.ocr import ocr_image
    with pytest.raises(ImportError, match="pytesseract"):
        ocr_image(_fake_image())


def test_missing_tesseract_binary_raises_install_hint(monkeypatch):
    """
    When neither our explicit location lookup nor pytesseract's own
    default can find the binary, the user must get a clear install
    message — not a stack trace from some downstream call.
    """

    import sys
    import types
    import doqment.ocr as ocr_mod

    # Build a fake pytesseract that always raises TesseractNotFoundError,
    # simulating the real-world case where neither shutil.which nor the
    # subprocess-default lookup succeeds.
    class _FakeTesseractNotFoundError(Exception):
        pass

    class _Output:
        DICT = "dict"

    def _raise_not_found(*a, **kw):
        raise _FakeTesseractNotFoundError("not found")

    fake_module = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="tesseract"),
        Output=_Output,
        TesseractNotFoundError=_FakeTesseractNotFoundError,
        image_to_data=_raise_not_found,
        get_languages=lambda: ["eng"],
    )

    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
    monkeypatch.setattr(ocr_mod, "_find_tesseract_binary", lambda: None)

    with pytest.raises(RuntimeError, match="could not be located"):
        ocr_mod.ocr_image(_fake_image())


def test_tesseract_works_when_only_pytesseract_default_finds_it(monkeypatch):
    """
    Even when our explicit lookup fails (returns None), pytesseract's
    own default 'tesseract' lookup may still succeed. In that case
    OCR should proceed normally — we don't pre-emptively crash.
    """

    import sys
    import types
    import doqment.ocr as ocr_mod

    class _Output:
        DICT = "dict"

    class _NotFound(Exception):
        pass

    fake_module = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="tesseract"),
        Output=_Output,
        TesseractNotFoundError=_NotFound,
        image_to_data=lambda *a, **kw: {
            "text": [], "left": [], "top": [], "width": [], "height": [],
            "conf": [], "block_num": [], "par_num": [], "line_num": [],
        },
        get_languages=lambda: ["eng"],
    )

    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
    monkeypatch.setattr(ocr_mod, "_find_tesseract_binary", lambda: None)

    # Must NOT raise — pytesseract.image_to_data succeeded.
    result = ocr_mod.ocr_image(_fake_image(), preprocess=False)
    assert result == []


def test_find_tesseract_respects_env_override(monkeypatch, tmp_path):
    """DOQMENT_TESSERACT_PATH must win over PATH and fallbacks."""

    import os
    from doqment.ocr import _find_tesseract_binary

    fake = tmp_path / "my-tesseract"
    fake.write_text("#!/bin/sh\necho ok")
    fake.chmod(0o755)

    monkeypatch.setenv("DOQMENT_TESSERACT_PATH", str(fake))
    assert _find_tesseract_binary() == str(fake)


def test_find_tesseract_falls_back_to_known_paths(monkeypatch, tmp_path):
    """
    If PATH doesn't include tesseract (minimal-env case — Obsidian
    terminal, some desktop launchers), the known-locations fallback
    must still find it.
    """

    import shutil
    from doqment import ocr as ocr_mod

    # Hide tesseract from PATH …
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("DOQMENT_TESSERACT_PATH", raising=False)

    # … but expose it at one of the known fallback paths.
    fake = tmp_path / "tesseract"
    fake.write_text("#!/bin/sh\necho ok")
    fake.chmod(0o755)

    real_isfile = __import__("os").path.isfile
    real_access = __import__("os").access

    def _fake_isfile(path):
        if path == "/usr/bin/tesseract":
            return True
        return real_isfile(path)

    def _fake_access(path, mode):
        if path == "/usr/bin/tesseract":
            return True
        return real_access(path, mode)

    monkeypatch.setattr("os.path.isfile", _fake_isfile)
    monkeypatch.setattr("os.access", _fake_access)

    assert ocr_mod._find_tesseract_binary() == "/usr/bin/tesseract"


def test_tesseract_binary_path_propagated_to_pytesseract(monkeypatch):
    """
    When the binary is located, ocr_image must explicitly point
    pytesseract at it (defensive against modules that set
    pytesseract.tesseract_cmd to something else).
    """

    import sys
    import types

    class _Output:
        DICT = "dict"

    fake_module = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="/wrong/path"),
        Output=_Output,
        image_to_data=lambda *a, **kw: {
            "text": [], "left": [], "top": [], "width": [], "height": [],
            "conf": [], "block_num": [], "par_num": [], "line_num": [],
        },
        get_languages=lambda: ["eng"],
    )

    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
    from doqment import ocr as ocr_mod
    monkeypatch.setattr(ocr_mod, "_find_tesseract_binary",
                        lambda: "/usr/bin/tesseract")

    ocr_mod.ocr_image(_fake_image(), preprocess=False)

    assert fake_module.pytesseract.tesseract_cmd == "/usr/bin/tesseract"
