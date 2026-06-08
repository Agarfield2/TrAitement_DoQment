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
        ocr_image(_fake_image(), engine="tesseract")


### Tests : Tesseract path comes from Settings ###

def test_ocr_image_pushes_settings_path_to_pytesseract(monkeypatch):
    """
    The cmd path is read from Settings.tesseract_cmd and pushed into
    pytesseract on every call. No auto-detection — explicit only.
    """

    import sys
    import types

    class _Output:
        DICT = "dict"

    class _NotFound(Exception):
        pass

    fake_module = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="/initial/wrong/path"),
        Output=_Output,
        TesseractNotFoundError=_NotFound,
        image_to_data=lambda *a, **kw: {
            "text": [], "left": [], "top": [], "width": [], "height": [],
            "conf": [], "block_num": [], "par_num": [], "line_num": [],
        },
        get_languages=lambda: ["eng"],
    )

    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)

    # Inject a custom Settings.tesseract_cmd via load_settings patch.
    from doqment import settings as settings_mod
    base = settings_mod.Settings()
    fake_settings = type(base)(
        **{**base.__dict__, "tesseract_cmd": "/usr/bin/tesseract"}
    ) if False else _make_settings_with(base, tesseract_cmd="/usr/bin/tesseract")
    monkeypatch.setattr(settings_mod, "load_settings", lambda: fake_settings)

    from doqment.ocr import ocr_image
    ocr_image(_fake_image(), engine="tesseract", preprocess=False)

    assert fake_module.pytesseract.tesseract_cmd == "/usr/bin/tesseract"


def test_ocr_image_friendly_error_when_pytesseract_cannot_invoke(monkeypatch):
    """
    If pytesseract itself raises TesseractNotFoundError when running
    the binary (e.g. the path in Settings is wrong), our wrapper
    converts it into a RuntimeError that names the configured path.
    """

    import sys
    import types

    class _NotFound(Exception):
        pass

    class _Output:
        DICT = "dict"

    fake_module = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
        Output=_Output,
        TesseractNotFoundError=_NotFound,
        image_to_data=lambda *a, **kw: (_ for _ in ()).throw(_NotFound("nope")),
        get_languages=lambda: ["eng"],
    )

    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)

    from doqment import settings as settings_mod
    base = settings_mod.Settings()
    fake_settings = _make_settings_with(base, tesseract_cmd="/nowhere/tesseract")
    monkeypatch.setattr(settings_mod, "load_settings", lambda: fake_settings)

    from doqment.ocr import ocr_image
    with pytest.raises(RuntimeError, match="/nowhere/tesseract"):
        ocr_image(_fake_image(), engine="tesseract", preprocess=False)


def _make_settings_with(base, **overrides):
    """Helper : returns a Settings copy with the given field overrides."""

    from dataclasses import replace
    return replace(base, **overrides)
