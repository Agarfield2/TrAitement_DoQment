"""
CLI for Pipeline 1 — textual RAG.

Three subcommands :

    ingest    Build the FAISS index from a folder of SROIE documents.
    doc       Q&A on a single document, no persistent index.
    db        Q&A across the indexed database.

LLM backend : Ollama, model from doqment/settings.py (default mistral:7b-instruct).
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import typer   # noqa: E402
from rich.console import Console   # noqa: E402

from doqment import phase1   # noqa: E402


app = typer.Typer(help=__doc__, add_completion=False)
console = Console()


@app.command()
def ingest(
    task1: str = typer.Option(
        "data/SROIE-Dataset_v2",
        "--task1", "--dir",
        help="Folder to ingest (scanned recursively).",
    ),
    task2: Optional[str] = typer.Option(
        None,
        help="Explicit folder of SROIE Task 2 entity JSONs. "
             "When omitted, entities are auto-discovered.",
    ),
    max_docs: Optional[int] = typer.Option(
        None, help="Cap on number of documents indexed.",
    ),
    use_ocr: bool = typer.Option(
        False, "--ocr/--no-ocr",
        help="Run OCR on images that lack a SROIE annotation. "
             "Default : skip those images.",
    ),
    ocr_engine: Optional[str] = typer.Option(
        None, "--ocr-engine",
        help="OCR engine: 'doctr' (default) or 'tesseract'. "
             "When omitted, uses Settings.ocr_engine.",
    ),
):
    """
    Builds the FAISS index from a folder of documents.
    """

    stats = phase1.ingest_directory(
        task1, task2, max_docs=max_docs, use_ocr=use_ocr,
        ocr_engine=ocr_engine,
    )
    console.print(f"[bold green]✓[/] Indexed {stats['total_passages']} passages "
                  f"from {stats['total_documents']} documents.")


@app.command()
def doc(
    file: str = typer.Option(..., help="Path to a single PDF or image."),
    question: str = typer.Option(..., help="Question to ask about that document."),
    annotation: Optional[str] = typer.Option(
        None, help="Optional SROIE annotation .txt file (skips OCR).",
    ),
    top_k: int = typer.Option(5, help="Passages kept for the prompt."),
):
    """
    Answers a question about ONE document, in memory, no database.
    """

    answer = phase1.ask_document(
        file=file, question=question, annotation=annotation, top_k=top_k,
    )
    _print(question, answer)


@app.command()
def db(
    question: str = typer.Option(..., help="Question to ask against the database."),
    top_k: int = typer.Option(5, help="Passages to retrieve."),
):
    """
    Answers a question using the whole indexed database.
    """

    answer = phase1.ask_database(question=question, top_k=top_k)
    _print(question, answer)


def _print(question, answer):
    """
    Pretty-prints the question, answer and supporting sources.

    Args:
        question (str): The original user question.
        answer (phase1.Answer): The pipeline output.
    """

    sep = "=" * 70
    console.print(f"\n[bold]{sep}[/]")
    console.print(f"[bold cyan]Question[/] : {question}")
    console.print(f"[bold]{sep}[/]\n")
    console.print(f"[bold green]Answer[/] :\n{answer.text}\n")
    console.print(f"[bold]Sources[/] (top {len(answer.sources)}) :")
    for i, src in enumerate(answer.sources, 1):
        console.print(
            f"  [{i}] [dim]{src.document} p.{src.page} "
            f"score={src.score:.3f}[/]\n      {src.snippet[:120]}"
        )


if __name__ == "__main__":
    app()
