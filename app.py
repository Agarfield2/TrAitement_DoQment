"""
TrAitement-DoQment — interface Streamlit.

Expose les trois modes (doc, db, ingest) pour les deux pipelines via
un sélecteur dans la barre latérale. Aucun appel réseau — tout tourne
sur la machine locale.
"""

import base64
import logging
import pathlib
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# page config
st.set_page_config(
    page_title="TrAitement-DoQment",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# helpers
def _logo_b64() -> str:
    """Return base64-encoded logo if the file exists, else empty string."""
    logo_path = pathlib.Path(__file__).parent / "logo_TDQ.png"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return ""


# global CSS
def _inject_css() -> None:
    logo_b64 = _logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{logo_b64}" '
        'style="width:48px;height:48px;border-radius:12px;object-fit:cover;" '
        'alt="TDQ logo" />'
        if logo_b64
        else '<div style="width:48px;height:48px;border-radius:12px;'
             'background:#1565C0;display:flex;align-items:center;justify-content:center;'
             'font-weight:900;font-size:14px;color:white;">TDQ</div>'
    )

    st.markdown(
        f"""
        <style>
        /* Google font ───────────────────────────────────*/
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700&display=swap');

        /* Reset / global ────────────────────────────────*/
        html, body, [class*="css"] {{
            font-family: 'DM Sans', sans-serif !important;
        }}

        /* Palette variables ─────────────────────────────*/
        :root {{
            --blue:        #1565C0;
            --blue-mid:    #1976D2;
            --blue-light:  #E3F2FD;
            --blue-soft:   #BBDEFB;
            --green:       #2E7D32;
            --green-mid:   #388E3C;
            --green-light: #E8F5E9;
            /* Theme-neutral tokens: legible on both light and dark surfaces.
               Page / widget backgrounds and body text are painted by the
               active Streamlit theme (see .streamlit/config.toml), so they are
               intentionally NOT hardcoded here anymore. */
            --border:      rgba(120,130,150,0.28);
            --muted:       #7C8798;
            --radius:      14px;
            --radius-sm:   8px;
        }}

        /* App background is painted by the active Streamlit theme. */

        /* Sidebar ───────────────────────────────────────*/
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0D47A1 0%, #1565C0 50%, #1B5E20 100%) !important;
        }}
        [data-testid="stSidebar"] > div {{
            background: transparent !important;
        }}
        /* all sidebar text white */
        [data-testid="stSidebar"] *:not(button) {{
            color: rgba(255,255,255,0.9) !important;
        }}

        /* radio labels */
        [data-testid="stSidebar"] .stRadio label {{
            color: rgba(255,255,255,0.75) !important;
            font-size: 13px !important;
            padding: 8px 10px;
            border-radius: 10px;
            transition: background 0.18s ease;
            cursor: pointer;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            background: rgba(255,255,255,0.10) !important;
            color: white !important;
        }}
        /* selected radio */
        [data-testid="stSidebar"] .stRadio [data-checked="true"] label,
        [data-testid="stSidebar"] .stRadio input:checked + label {{
            background: rgba(255,255,255,0.18) !important;
            color: white !important;
            font-weight: 500 !important;
        }}

        /* radio circle accent */
        [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] svg {{
            fill: #69F0AE !important;
        }}

        /* divider */
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.15) !important;
        }}
        /* caption */
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] small {{
            color: rgba(255,255,255,0.45) !important;
            font-size: 11px !important;
        }}

        /* Main content area ─────────────────────────────*/
        .block-container {{
            padding: 2rem 2.5rem 2rem !important;
            max-width: 900px;
        }}

        /* Page title ────────────────────────────────────*/
        h1 {{
            font-family: 'Syne', sans-serif !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }}
        h2 {{
            font-family: 'Syne', sans-serif !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
        }}
        h3 {{
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--muted) !important;
            margin-bottom: 0.5rem;
        }}

        /* Buttons ───────────────────────────────────────*/
        .stButton > button {{
            background: linear-gradient(135deg, var(--blue) 0%, var(--blue-mid) 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.55rem 1.4rem !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            box-shadow: 0 2px 8px rgba(21,101,192,0.28) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 18px rgba(21,101,192,0.38) !important;
        }}
        .stButton > button:active {{
            transform: translateY(0) !important;
        }}

        /* File uploader ─────────────────────────────────*/
        [data-testid="stFileUploader"] {{
            border: 2px dashed var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 1.5rem !important;
            transition: border-color 0.2s, background 0.2s !important;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: var(--blue-mid) !important;
            background: rgba(21,101,192,0.08) !important;
        }}

        /* Text inputs / text areas ──────────────────────*/
        .stTextArea textarea,
        .stTextInput input {{
            border: 1.5px solid var(--border) !important;
            border-radius: 10px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 14px !important;
            transition: border-color 0.18s ease !important;
        }}
        .stTextArea textarea:focus,
        .stTextInput input:focus {{
            border-color: var(--blue-mid) !important;
            box-shadow: 0 0 0 3px rgba(25,118,210,0.12) !important;
        }}

        /* Select boxes ──────────────────────────────────*/
        .stSelectbox > div > div {{
            border-radius: 10px !important;
            border: 1.5px solid var(--border) !important;
            font-size: 13px !important;
        }}

        /* Expanders ─────────────────────────────────────*/
        .stExpander {{
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
        }}

        /* Info / success / warning boxes ────────────────*/
        .stAlert {{
            border-radius: var(--radius) !important;
            border: none !important;
        }}

        /* Spinner ───────────────────────────────────────*/
        .stSpinner > div {{
            border-top-color: var(--blue) !important;
        }}

        /* Fade-in animation for content ─────────────────*/
        @keyframes tdqFadeUp {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .block-container > div > div > div {{
            animation: tdqFadeUp 0.35s ease both;
        }}

        /* Status badge (pipeline) ───────────────────────*/
        .tdq-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(21,101,192,0.14);
            color: #3B82F6;
            border: 1px solid rgba(59,130,246,0.30);
            font-size: 11px;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            letter-spacing: 0.03em;
        }}
        .tdq-badge.green {{
            background: rgba(46,125,50,0.16);
            color: #4CAF50;
            border-color: rgba(76,175,80,0.30);
        }}
        .tdq-badge-dot {{
            width: 7px; height: 7px;
            border-radius: 50%;
            background: currentColor;
            animation: tdqPulse 2s infinite;
        }}
        @keyframes tdqPulse {{
            0%,100% {{ opacity:1; }} 50% {{ opacity:0.35; }}
        }}

        /* Answer block ──────────────────────────────────*/
        .tdq-answer {{
            background: rgba(127,127,140,0.06);
            border-left: 3px solid var(--green-mid);
            border-top: 1px solid var(--border);
            border-right: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
            border-radius: 0 var(--radius) var(--radius) 0;
            padding: 1.2rem 1.4rem;
            margin-top: 1rem;
            animation: tdqFadeUp 0.4s ease both;
        }}
        .tdq-answer-label {{
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #4CAF50;
            margin-bottom: 8px;
        }}

        /* Cards ─────────────────────────────────────────*/
        .tdq-card {{
            background: rgba(127,127,140,0.06);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }}

        /* Pipeline header bar ───────────────────────────*/
        .tdq-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
        }}
        </style>

        <!-- Sidebar logo + branding injected via markdown -->
        """,
        unsafe_allow_html=True,
    )

    # Inject logo into sidebar top
    st.sidebar.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;padding:4px 0 16px;">
            {logo_html}
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                            color:white;line-height:1.2;">TrAitement-DoQment</div>
                <div style="font-size:11px;color:rgba(255,255,255,0.45);margin-top:2px;">
                    Local document intelligence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# sidebar
def render_sidebar() -> tuple[str, str]:
    """
    Renders the navigation sidebar.

    Returns:
        tuple: (pipeline_code, mode_code).
            pipeline_code → "phase1" | "phase2"
            mode_code     → "doc" | "db" | "ingest"
    """
    with st.sidebar:
        st.divider()

        st.markdown("### Pipeline")
        pipeline = st.radio(
            "Pipeline",
            options=["Pipeline 1 — Textual", "Pipeline 2 — Multimodal"],
            key="pipeline",
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("### Mode")
        mode = st.radio(
            "Mode",
            options=[
                "📄  Doc — one file + a question",
                "🗂️  BD — question across the database",
                "⚙️  Ingestion — (re)build the database",
            ],
            key="mode",
            label_visibility="collapsed",
        )

        st.divider()

        # Backend status badge
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:8px;
                        background:rgba(255,255,255,0.08);border-radius:8px;padding:8px 10px;">
                <div style="width:7px;height:7px;border-radius:50%;background:#69F0AE;
                            animation:tdqPulse 2s infinite;flex-shrink:0;"></div>
                <div style="font-size:11px;color:rgba(255,255,255,0.5);">
                    LLM via <strong style="color:rgba(255,255,255,0.8);">Ollama</strong>
                    · configured in <code style="font-size:10px;
                    color:rgba(255,255,255,0.6);">settings.py</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    pipeline_code = "phase1" if pipeline.startswith("Pipeline 1") else "phase2"
    mode_code = (
        "doc" if mode.startswith("📄")
        else "db" if mode.startswith("🗂️")
        else "ingest"
    )
    return pipeline_code, mode_code


# page header
_MODE_META = {
    "doc": {
        "title": "Doc Mode",
        "subtitle": "Upload a single document and ask a question.",
        "badge_class": "tdq-badge",
    },
    "db": {
        "title": "BD Mode",
        "subtitle": "Ask a question against your indexed document database.",
        "badge_class": "tdq-badge",
    },
    "ingest": {
        "title": "Ingestion",
        "subtitle": "Rebuild or update your document index.",
        "badge_class": "tdq-badge green",
    },
}

_PIPELINE_LABEL = {
    "phase1": "Pipeline 1 · Textual",
    "phase2": "Pipeline 2 · Multimodal",
}


def render_page_header(pipeline_code: str, mode_code: str) -> None:
    meta = _MODE_META[mode_code]
    badge_cls = meta["badge_class"]
    pipeline_label = _PIPELINE_LABEL[pipeline_code]

    st.markdown(
        f"""
        <div class="tdq-header">
            <div>
                <h1 style="margin:0 0 4px;">{meta['title']}</h1>
                <p style="margin:0;font-size:13px;color:var(--muted);">{meta['subtitle']}</p>
            </div>
            <div class="{badge_cls}">
                <span class="tdq-badge-dot"></span>
                {pipeline_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# router
def main() -> None:
    _inject_css()

    pipeline_code, mode_code = render_sidebar()
    render_page_header(pipeline_code, mode_code)


    from doqment.ui import doc_view, db_view, ingest_view

    if mode_code == "doc":
        doc_view.render(pipeline_code)
    elif mode_code == "db":
        db_view.render(pipeline_code)
    else:
        ingest_view.render(pipeline_code)


if __name__ == "__main__":
    main()
