"""
Streamlit application for TrAitement-DoQment.

Sidebar lets the user pick :
  - Pipeline : 1 (textual) or 2 (multimodal)
  - Mode    : Doc (one file + question), BD (question against the index),
              Ingest ((re)build the index)

Heavy models are cached via st.cache_resource so the first launch pays
the warm-up cost once.
"""

import logging

import streamlit as st


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


st.set_page_config(
    page_title="TrAitement-DoQment",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


### Sidebar ###

def render_sidebar():
    """
    Renders the navigation sidebar.

    Returns:
        tuple: (pipeline_code, mode_code). pipeline_code is "phase1" or
            "phase2"; mode_code is "doc", "db" or "ingest".
    """

    with st.sidebar:
        st.markdown("# 📄 TrAitement-DoQment")
        st.caption("100% local document search.")
        st.divider()

        pipeline = st.radio(
            "**Pipeline**",
            options=["Pipeline 1 (textual)", "Pipeline 2 (multimodal)"],
            key="pipeline",
        )
        mode = st.radio(
            "**Mode**",
            options=[
                "📄 Doc — one document + a question",
                "🗂️ BD — question across the database",
                "⚙️ Ingestion — (re)build the database",
            ],
            key="mode",
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("LLM backend : Ollama (configured in `doqment/settings.py`).")

    pipeline_code = "phase1" if pipeline.startswith("Pipeline 1") else "phase2"
    mode_code = "doc" if mode.startswith("📄") else "db" if mode.startswith("🗂️") else "ingest"
    return pipeline_code, mode_code


### Router ###

def main():
    """
    Dispatches rendering based on the sidebar selectors.
    """

    pipeline_code, mode_code = render_sidebar()
    from doqment.ui import doc_view, db_view, ingest_view

    if mode_code == "doc":
        doc_view.render(pipeline_code)
    elif mode_code == "db":
        db_view.render(pipeline_code)
    else:
        ingest_view.render(pipeline_code)


if __name__ == "__main__":
    main()
