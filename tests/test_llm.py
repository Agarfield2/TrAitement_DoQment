"""
Tests for the Ollama LLM client.

We do not call Ollama in tests — they would need a daemon and a
~5 Go model. Instead we exercise the response-parsing layer, which
is where the bugs would actually live.
"""

import json

import pytest

from doqment import llm


### Tests : VLM JSON parsing ###

def test_parse_valid_json_extracts_answer_and_cites():
    raw = json.dumps({"answer": "Le total est 42.50.", "cited_pages": [1, 3]})
    result = llm._parse_vlm_response(raw, n_pages=5)
    assert result.answer == "Le total est 42.50."
    assert result.cited_pages == [1, 3]


def test_parse_drops_out_of_range_indices():
    raw = json.dumps({"answer": "x", "cited_pages": [0, 1, 6, -1, 3]})
    result = llm._parse_vlm_response(raw, n_pages=5)
    assert result.cited_pages == [1, 3]


def test_parse_dedups_repeated_indices():
    raw = json.dumps({"answer": "x", "cited_pages": [1, 1, 2, 2, 1]})
    result = llm._parse_vlm_response(raw, n_pages=5)
    assert result.cited_pages == [1, 2]


def test_parse_handles_non_integer_indices():
    raw = json.dumps({"answer": "x", "cited_pages": ["1", "two", None, 2]})
    result = llm._parse_vlm_response(raw, n_pages=5)
    assert result.cited_pages == [1, 2]


def test_parse_handles_missing_cited_pages_key():
    raw = json.dumps({"answer": "ok"})
    result = llm._parse_vlm_response(raw, n_pages=5)
    assert result.cited_pages == []


def test_parse_handles_invalid_json_gracefully():
    raw = "I am not JSON {{{"
    result = llm._parse_vlm_response(raw, n_pages=5)
    assert result.answer == raw
    assert result.cited_pages == []


def test_parse_handles_non_list_cited_pages():
    raw = json.dumps({"answer": "x", "cited_pages": "not a list"})
    result = llm._parse_vlm_response(raw, n_pages=5)
    assert result.cited_pages == []


### Tests : image encoding ###

def test_image_to_b64_produces_valid_base64():
    import base64
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="red")
    encoded = llm._image_to_b64(img)
    decoded = base64.b64decode(encoded)
    assert decoded.startswith(b"\x89PNG"), "Output should be PNG bytes"


def test_image_to_b64_converts_non_rgb():
    """RGBA, L, P modes must be silently converted to RGB."""
    from PIL import Image

    for mode in ("RGBA", "L", "P"):
        img = Image.new(mode, (10, 10))
        encoded = llm._image_to_b64(img)
        assert isinstance(encoded, str) and len(encoded) > 0


### Tests : missing dependency ###

def test_client_raises_clear_error_when_ollama_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ollama", None)
    with pytest.raises(ImportError, match="ollama"):
        llm._client("http://localhost:11434")
