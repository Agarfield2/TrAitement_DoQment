"""
Tests for the Ollama LLM client.

We do not call Ollama in tests — they would need a daemon and a
~5 Go model. Instead we stub the client and exercise the thin
request/response layer.
"""

import pytest

from doqment import llm


### Tests : vision generation (no JSON envelope) ###

def test_generate_vision_returns_model_prose(monkeypatch):
    """generate_vision returns the model's text content, stripped, with no
    JSON format constraint imposed on the model."""
    from PIL import Image

    captured = {}

    class FakeClient:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"message": {"content": "  Le total est 42,50.  "}}

    monkeypatch.setattr(llm, "_client", lambda host: FakeClient())

    out = llm.generate_vision(
        prompt="Quel est le total ?",
        images=[Image.new("RGB", (10, 10))],
        model="qwen2.5vl:7b", host="http://x",
    )
    assert out == "Le total est 42,50."
    assert "format" not in captured            # plus d'enveloppe JSON forcée
    assert captured["options"]["num_ctx"] >= 1  # garde-fous contexte toujours là


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


def test_image_to_b64_downscales_to_max_side():
    """max_side caps the longest dimension while keeping aspect ratio."""
    import base64
    import io
    from PIL import Image

    img = Image.new("RGB", (2000, 1000), color="blue")
    encoded = llm._image_to_b64(img, max_side=1024)
    out = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert max(out.size) == 1024
    assert out.size == (1024, 512)


### Tests : missing dependency ###

def test_client_raises_clear_error_when_ollama_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ollama", None)
    with pytest.raises(ImportError, match="ollama"):
        llm._client("http://localhost:11434")
