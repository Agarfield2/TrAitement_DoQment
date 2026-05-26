"""
Tests for the Typer CLIs in scripts/.

These guard against a silly-but-deadly mistake : if a Typer Option
has no type annotation, Typer treats it as `str`, and the string
"False" coerces to truthy `bool` downstream. We hit this exact bug
on the `--paddle` flag of `phase1 ingest`. These tests catch it.
"""

from typer.testing import CliRunner


### scripts/phase1.py ###

def test_phase1_ingest_use_tesseract_is_a_bool_flag():
    """No `--tesseract` flag → use_tesseract is the boolean `False`."""

    import scripts.phase1 as cli

    runner = CliRunner()
    # Build a tiny shim that captures the parsed value.
    captured = {}

    def _shim(task1, task2, **kw):
        captured.update(kw)
        return {"total_passages": 0, "total_documents": 0}

    import doqment.phase1 as phase1_mod
    phase1_mod_ingest = phase1_mod.ingest_directory
    phase1_mod.ingest_directory = _shim
    try:
        result = runner.invoke(cli.app, ["ingest"])
    finally:
        phase1_mod.ingest_directory = phase1_mod_ingest

    assert result.exit_code == 0, result.output
    # The actual value must be the boolean False — not "False".
    assert captured["use_tesseract"] is False


def test_phase1_ingest_with_tesseract_flag_sets_true():
    import scripts.phase1 as cli

    runner = CliRunner()
    captured = {}

    def _shim(task1, task2, **kw):
        captured.update(kw)
        return {"total_passages": 0, "total_documents": 0}

    import doqment.phase1 as phase1_mod
    real = phase1_mod.ingest_directory
    phase1_mod.ingest_directory = _shim
    try:
        result = runner.invoke(cli.app, ["ingest", "--tesseract"])
    finally:
        phase1_mod.ingest_directory = real

    assert result.exit_code == 0, result.output
    assert captured["use_tesseract"] is True


def test_phase1_doc_top_k_is_int():
    """--top-k 7 must come through as the int 7, not the string '7'."""

    import scripts.phase1 as cli

    runner = CliRunner()
    captured = {}

    def _shim(**kw):
        captured.update(kw)
        return type("A", (), {"text": "", "sources": []})()

    import doqment.phase1 as phase1_mod
    real = phase1_mod.ask_document
    phase1_mod.ask_document = _shim
    try:
        result = runner.invoke(cli.app, [
            "doc", "--file", "x.png", "--question", "?", "--top-k", "7",
        ])
    finally:
        phase1_mod.ask_document = real

    assert result.exit_code == 0, result.output
    assert captured["top_k"] == 7
    assert isinstance(captured["top_k"], int)


### scripts/phase2.py ###

def test_phase2_db_retrieve_and_generate_k_are_ints():
    import scripts.phase2 as cli

    runner = CliRunner()
    captured = {}

    def _shim(**kw):
        captured.update(kw)
        return type("A", (), {"text": "", "sources": [], "cited": []})()

    import doqment.phase2 as phase2_mod
    real = phase2_mod.ask_database
    phase2_mod.ask_database = _shim
    try:
        result = runner.invoke(cli.app, [
            "db", "--question", "?",
            "--retrieve-k", "15", "--generate-k", "4",
        ])
    finally:
        phase2_mod.ask_database = real

    assert result.exit_code == 0, result.output
    assert captured == {"question": "?", "retrieve_k": 15, "generate_k": 4}
