"""
DB-mode view : question against the whole indexed database.

The user types a question, the selected pipeline retrieves across
every indexed document and answers with citations.
"""

import streamlit as st


### Public API ###

def render(pipeline_code):
    """
    Renders the db-mode view for the selected pipeline.

    Args:
        pipeline_code (str): "phase1" or "phase2".
    """

    st.title("🗂️ BD mode — ask the whole database")

    _warn_if_missing(pipeline_code)

    question = st.text_input("Question", key=f"q_db_{pipeline_code}")

    if pipeline_code == "phase1":
        top_k = st.slider("Top-k passages", 1, 20, 5, key="db_k_p1")
    else:
        col1, col2 = st.columns(2)
        retrieve_k = col1.slider(
            "retrieve_k (Qdrant)", 1, 30, 10, key="db_ret_k",
        )
        generate_k = col2.slider(
            "generate_k (sent to VLM)", 1, 10, 3, key="db_gen_k",
        )

    submit = st.button(
        "🔎 Ask", type="primary",
        disabled=not question.strip(),
        key=f"submit_db_{pipeline_code}",
    )
    if not submit:
        return

    with st.spinner("Searching ..."):
        try:
            if pipeline_code == "phase1":
                _run_phase1(question, top_k)
            else:
                _run_phase2(question, retrieve_k, generate_k)
        except Exception as exc:
            st.error(f"Pipeline error : {exc}")
            st.exception(exc)


### Index presence ###

def _warn_if_missing(pipeline_code):
    """
    Renders a warning if the chosen pipeline has no index yet.
    """

    from doqment.settings import load_settings

    settings = load_settings()
    if pipeline_code == "phase1":
        idx = settings.phase1_index_dir / "index.faiss"
        if not idx.exists():
            st.warning(
                f"⚠️ Phase 1 index not found at `{idx}`. Build it via "
                "the Ingestion mode or `python scripts/phase1.py ingest`."
            )
    else:
        qdr = settings.phase2_qdrant_dir
        if not qdr.exists():
            st.warning(
                f"⚠️ Phase 2 Qdrant database not found at `{qdr}`. Build it "
                "via the Ingestion mode or `python scripts/phase2.py ingest`."
            )


### Phase 1 ###

def _run_phase1(question, top_k):
    """
    Calls Pipeline 1 ask_database() and renders the answer + sources.
    """

    from doqment import phase1

    answer = phase1.ask_database(
        question=question, top_k=top_k,
        embedder=_get_phase1_embedder(),
    )

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

def _run_phase2(question, retrieve_k, generate_k):
    """
    Calls Pipeline 2 ask_database() and renders the answer + pages.
    """

    from doqment import phase2

    answer = phase2.ask_database(
        question=question, retrieve_k=retrieve_k, generate_k=generate_k,
        encoder=_get_phase2_encoder(),
    )

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
