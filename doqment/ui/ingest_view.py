"""
Ingestion-mode view : (re)build the database for the selected pipeline.
"""

from pathlib import Path

import streamlit as st


### Public API ###

def render(pipeline_code):
    """
    Renders the ingest-mode view for the selected pipeline.

    Args:
        pipeline_code (str): "phase1" or "phase2".
    """

    st.title("⚙️ Ingestion — build the database")

    if pipeline_code == "phase1":
        _render_phase1()
    else:
        _render_phase2()


### Phase 1 ###

def _render_phase1():
    """
    Renders the Phase 1 ingestion form.
    """

    st.caption(
        "Phase 1 : scans the folder **recursively**, uses SROIE-style "
        "annotations (`.txt` next to each image) and auto-discovers "
        "entity JSON files anywhere in the tree. Output goes to "
        "`data/processed/`."
    )

    root = st.text_input(
        "Folder to ingest (subfolders included)",
        value="data/SROIE-Dataset_v2",
    )

    with st.expander("Advanced options"):
        task2 = st.text_input(
            "Explicit entities folder (leave empty for auto-detect)",
            value="",
        )
        max_docs = st.number_input(
            "Cap on documents (0 = no cap)", min_value=0, value=0, step=10,
        )
        use_ocr = st.checkbox(
            "Run OCR on images without annotation "
            "(otherwise they are skipped)",
            value=False,
        )
        ocr_engine = st.selectbox(
            "OCR engine",
            options=["doctr", "tesseract"],
            index=0,
            help="docTR (default) is more accurate on receipts ; "
                 "Tesseract is the classic binary path.",
        )

    if not st.button("🚀 Build the Phase 1 index", type="primary"):
        return

    if not Path(root).exists():
        st.error(f"Folder does not exist : {root}")
        return

    with st.spinner("Ingesting … this can take a few minutes."):
        try:
            from doqment import phase1

            stats = phase1.ingest_directory(
                task1=root,
                task2=task2 or None,
                max_docs=int(max_docs) if max_docs else None,
                use_ocr=use_ocr,
                ocr_engine=ocr_engine,
            )
        except Exception as exc:
            st.error(f"Ingestion failed : {exc}")
            st.exception(exc)
            return

    st.success("✓ Phase 1 ingestion done.")
    st.json(stats)


### Phase 2 ###

def _render_phase2():
    """
    Renders the Phase 2 ingestion form (folder → Qdrant).
    """

    st.caption(
        "Phase 2 : pdf2image rasterization → ColQwen2 encoding → Qdrant "
        "+ SQLite. Idempotent via MD5 — already-indexed files are skipped."
    )

    src_dir = st.text_input("Source directory", value="data/SROIE-Dataset_v2")

    if not st.button("🚀 Build the Phase 2 index", type="primary"):
        return

    if not Path(src_dir).is_dir():
        st.error(f"Source directory does not exist : {src_dir}")
        return

    with st.spinner("Encoding pages with ColQwen2 … this can take a while."):
        try:
            from doqment import phase2

            n_pages = phase2.ingest_directory(src_dir)
        except Exception as exc:
            st.error(f"Ingestion failed : {exc}")
            st.exception(exc)
            return

    st.success(f"✓ Phase 2 ingestion done — {n_pages} new page(s) indexed.")
