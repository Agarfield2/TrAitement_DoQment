"""
Tests for doqment.settings.

The Settings dataclass is frozen and has no logic — the only thing
worth testing is that environment variables actually override the
defaults.
"""

import importlib


def test_default_settings_have_expected_models():
    from doqment.settings import load_settings

    s = load_settings()
    assert s.ollama_text_model.startswith("mistral")
    assert s.ollama_vision_model.startswith("qwen")
    assert s.ollama_host.startswith("http")


def test_env_var_overrides_text_model(monkeypatch):
    monkeypatch.setenv("DOQMENT_OLLAMA_TEXT_MODEL", "llama3.2:3b")
    import doqment.settings as settings
    importlib.reload(settings)
    s = settings.load_settings()
    assert s.ollama_text_model == "llama3.2:3b"
    monkeypatch.delenv("DOQMENT_OLLAMA_TEXT_MODEL", raising=False)
    importlib.reload(settings)   # restore for downstream tests


def test_env_var_overrides_host(monkeypatch):
    monkeypatch.setenv("DOQMENT_OLLAMA_HOST", "http://10.0.0.5:11434")
    import doqment.settings as settings
    importlib.reload(settings)
    s = settings.load_settings()
    assert s.ollama_host == "http://10.0.0.5:11434"
    monkeypatch.delenv("DOQMENT_OLLAMA_HOST", raising=False)
    importlib.reload(settings)
