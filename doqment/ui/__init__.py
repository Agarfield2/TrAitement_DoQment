"""
Streamlit UI views for TrAitement-DoQment.

Each module exposes a `render(pipeline_code)` function that draws one
mode (doc, db, ingest) for one pipeline (phase1, phase2). The dispatcher
in app.py decides which to call based on sidebar selections.
"""
