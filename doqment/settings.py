"""
Configuration for both pipelines.

A single dataclass holds every knob. Defaults are sensible for a fresh
checkout ; override by editing this file or by setting environment
variables of the form `DOQMENT_<FIELD>=value` before launching.

We deliberately do NOT use pydantic-settings + YAML : the project is
small enough that one dataclass with `os.environ.get` defaults is
clearer and faster to read.
"""

import os
from dataclasses import dataclass
from pathlib import Path


### Defaults ###

_DATA = Path("data")


@dataclass(frozen=True)
class Settings:
    """
    Project-wide settings.

    Attributes:
        ollama_host: URL of the Ollama daemon, e.g. http://localhost:11434.
        ollama_text_model: Model tag used for Pipeline 1 text answers.
        ollama_vision_model: Model tag used for Pipeline 2 multimodal answers.
        ollama_keep_alive: How long Ollama keeps the model warm in RAM/VRAM.
        ollama_num_ctx: Ollama context window for the VLM (token budget).
        phase1_index_dir: Directory holding FAISS index + metadata (Pipeline 1).
        phase2_qdrant_dir: Directory holding the local Qdrant database (Pipeline 2).
        phase2_metadata_db: SQLite file holding Pipeline 2 page metadata.
        phase2_pages_dir: Directory of rasterized page PNGs (Pipeline 2).
        phase2_raw_dir: Default input folder for Pipeline 2 ingestion.
        colqwen_model: Hugging Face name of the ColQwen2 encoder.
        colqwen_device: Torch device for ColQwen2 ("cuda:0", "cpu").
        colqwen_dtype: Torch dtype for ColQwen2 ("bfloat16", "float16", "float32").
        ocr_engine: Default OCR engine for Pipeline 1's fallback,
            "doctr" (default) or "tesseract".
        retrieve_k: Pages retrieved before sending to the VLM (Pipeline 2).
        generate_k: Pages actually sent to the VLM (Pipeline 2).
    """

    ollama_host: str = os.environ.get("DOQMENT_OLLAMA_HOST", "http://localhost:11434")
    ollama_text_model: str = os.environ.get("DOQMENT_OLLAMA_TEXT_MODEL", "mistral:7b-instruct")
    ollama_vision_model: str = os.environ.get("DOQMENT_OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
    ollama_keep_alive: str = os.environ.get("DOQMENT_OLLAMA_KEEP_ALIVE", "5m")
    # Fenêtre de contexte Ollama pour le VLM. Plusieurs images de pages
    # consomment beaucoup de tokens visuels ; une fenêtre trop petite tronque
    # l'entrée et fait dégénérer Qwen (boucle de tokens <|im_start|>).
    ollama_num_ctx: int = int(os.environ.get("DOQMENT_OLLAMA_NUM_CTX", "8192"))

    phase1_index_dir: Path = _DATA / "processed"
    phase2_qdrant_dir: Path = _DATA / "qdrant"
    phase2_metadata_db: Path = _DATA / "metadata.db"
    phase2_pages_dir: Path = _DATA / "pages"
    phase2_raw_dir: Path = _DATA / "raw"

    colqwen_model: str = os.environ.get("DOQMENT_COLQWEN_MODEL", "vidore/colqwen2-v1.0")
    colqwen_device: str = os.environ.get("DOQMENT_COLQWEN_DEVICE", "cpu")
    colqwen_dtype: str = os.environ.get("DOQMENT_COLQWEN_DTYPE", "bfloat16")

    # OCR engine for Pipeline 1's fallback (images without a SROIE
    # annotation). "doctr" (default) is a deep-learning recognizer, far
    # more accurate on receipts ; "tesseract" keeps the classic binary
    # path. Override via DOQMENT_OCR_ENGINE.
    ocr_engine: str = os.environ.get("DOQMENT_OCR_ENGINE", "doctr")

    # Tesseract binary — hardcoded to /usr/bin/tesseract because that's
    # where Fedora, Debian and Ubuntu put it. Override via the env var
    # DOQMENT_TESSERACT_PATH or by editing this line if your distro
    # installs it elsewhere (Arch : same path ; macOS Homebrew :
    # /opt/homebrew/bin/tesseract).
    tesseract_cmd: str = os.environ.get("DOQMENT_TESSERACT_PATH", "/usr/bin/tesseract")

    retrieve_k: int = 10
    generate_k: int = 3


def load_settings():
    """
    Returns the project Settings — a thin wrapper kept for clarity.

    Returns:
        Settings: A fresh frozen Settings instance.
    """

    return Settings()
