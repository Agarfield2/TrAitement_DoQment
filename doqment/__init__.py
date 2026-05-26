"""
TrAitement-DoQment — local Retrieval-Augmented Generation toolkit.

Two pipelines, one Ollama backend, no network access at run time.

- Pipeline 1 (textual)    : Tesseract OCR + MPNet 768d + FAISS HNSW + Mistral 7B.
- Pipeline 2 (multimodal) : ColQwen2 visual embeddings + Qdrant + Qwen2.5-VL 7B.

Public entry points :

    from doqment import phase1, phase2

    phase1.ingest_directory(...)
    phase1.ask_document(...)
    phase1.ask_database(...)

    phase2.ingest_directory(...)
    phase2.ask_document(...)
    phase2.ask_database(...)

See README.md for installation and CLI usage.
"""

__version__ = "1.0.0"
__author__ = "BEHAREL Armand, JEANNE Arthur, SOUKI Mohamed"
