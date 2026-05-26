"""
Doc-mode view : one document + one question.

The user uploads a PDF/image, types a question, the selected pipeline
answers. No persistent index is touched.
"""

import tempfile
from pathlib import Path

import streamlit as st


### Public API ###

def render(pipeline_code):
    """
    Renders the doc-mode view for the selected pipeline.

    Args:
        pipeline_code (str): "phase1" or "phase2".
    """

    st.title("📄 Doc mode — one document + one question")

    accepted = ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "pdf"]
    uploaded = st.file_uploader(
        "Upload a document", type=accepted, key=f"upload_{pipeline_code}",
    )

    annot = None
    if pipeline_code == "phase1":
        st.info(
            "💡 If you uploaded a SROIE receipt, upload the matching `.txt` "
            "annotation below to skip OCR. Otherwise Tesseract runs (requires "
            "the `tesseract` binary system-wide)."
        )
        annot = st.file_uploader(
            "Optional : SROIE annotation .txt", type=["txt"],
            key="upload_annot",
        )
    else:
        st.caption(
            "Pipeline 2 rasterizes the document and asks Qwen2.5-VL via "
            "Ollama. Requires Ollama running with `qwen2.5vl:7b` pulled."
        )

    question = st.text_input("Question", key=f"q_{pipeline_code}")
    top_k = st.slider(
        "Top-k passages / pages", 1, 10,
        value=5 if pipeline_code == "phase1" else 3,
        key=f"k_{pipeline_code}",
    )

    submit = st.button(
        "🔎 Ask", type="primary",
        disabled=not (uploaded and question.strip()),
        key=f"submit_{pipeline_code}",
    )
    if not submit:
        return

    with tempfile.TemporaryDirectory(prefix="doqment_") as tmp:
        tmp_dir = Path(tmp)
        file_path = tmp_dir / uploaded.name
        file_path.write_bytes(uploaded.getvalue())

        annot_path = None
        if annot is not None:
            annot_path = tmp_dir / annot.name
            annot_path.write_bytes(annot.getvalue())

        with st.spinner("Running the pipeline ..."):
            try:
                if pipeline_code == "phase1":
                    _run_phase1(file_path, question, annot_path, top_k)
                else:
                    _run_phase2(file_path, question, top_k)
            except Exception as exc:
                st.error(f"Pipeline error : {exc}")
                st.exception(exc)


### Phase 1 ###

def _run_phase1(file_path, question, annot_path, top_k):
    """
    Calls Pipeline 1 ask_document() and renders the answer + sources.
    """

    from doqment import phase1

    answer = phase1.ask_document(
        file=file_path, question=question,
        annotation=annot_path, top_k=top_k,
        embedder=_get_phase1_embedder(),
    )
    _render_text_answer(answer)


def _render_text_answer(answer):
    """
    Renders a phase1.Answer block.
    """

    st.subheader("Answer")
    st.write(answer.text)

    if not answer.sources:
        st.info("No source was retrieved.")
        return

    st.subheader(f"Sources (top {len(answer.sources)})")
    for i, src in enumerate(answer.sources, 1):
        with st.expander(
            f"[{i}] {src.document} p.{src.page} — score {src.score:.3f}",
            expanded=(i == 1),
        ):
            st.text(src.snippet)
            if src.entities:
                st.markdown("**Entities**")
                st.json(src.entities)


### Phase 2 ###

def _run_phase2(file_path, question, top_k):
    """
    Calls Pipeline 2 ask_document() and renders the answer + cited pages.
    """

    from doqment import phase2

    answer = phase2.ask_document(
        file=file_path, question=question, generate_k=top_k,
        encoder=_get_phase2_encoder(),
    )
    _render_vision_answer(answer)


def _render_vision_answer(answer):
    """
    Renders a phase2.Answer block with cited-page thumbnails.
    """

    st.subheader("Answer")
    st.write(answer.text)

    if answer.cited:
        st.subheader(f"Cited pages ({len(answer.cited)})")
        cols = st.columns(min(len(answer.cited), 3))
        for i, src in enumerate(answer.cited):
            with cols[i % len(cols)]:
                st.image(
                    str(src.image_path),
                    caption=f"{src.document} p.{src.page} — score {src.score:.3f}",
                    use_container_width=True,
                )

    if answer.sources:
        with st.expander(f"All retrieved pages ({len(answer.sources)})"):
            for src in answer.sources:
                st.markdown(
                    f"- **{src.document}** p.{src.page} — score `{src.score:.3f}`"
                )


### Cached models ###

@st.cache_resource(show_spinner="Loading MPNet embedder ...")
def _get_phase1_embedder():
    """
    Returns a cached MPNet embedder (Phase 1).
    """

    from ingestion import EmbeddingModel

    embedder = EmbeddingModel()
    embedder._load()
    return embedder


@st.cache_resource(show_spinner="Loading ColQwen2 encoder ...")
def _get_phase2_encoder():
    """
    Returns a cached ColQwen2 encoder (Phase 2).
    """

    from doqment import phase2_store
    from doqment.settings import load_settings

    s = load_settings()
    return phase2_store.ColQwen2Encoder(
        model_name=s.colqwen_model,
        device=s.colqwen_device,
        dtype=s.colqwen_dtype,
    )
