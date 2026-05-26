"""
CLI for Pipeline 2 — multimodal RAG.

Three subcommands :

    ingest    Index a folder of PDFs / images into Qdrant.
    doc       Q&A on a single document, no persistent index.
    db        Q&A across the indexed database.

Generation backend : Ollama, model from doqment/settings.py (default qwen2.5vl:7b).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import typer   # noqa: E402
from rich.console import Console   # noqa: E402

from doqment import phase2   # noqa: E402


app = typer.Typer(help=__doc__, add_completion=False)
console = Console()


@app.command()
def ingest(
    dir: str = typer.Option("data/raw", help="Folder of PDFs / images to index."),
):
    """
    Indexes a folder into the persistent Qdrant database.
    """

    n_pages = phase2.ingest_directory(dir)
    console.print(f"[bold green]✓[/] Indexed {n_pages} new page(s).")


@app.command()
def doc(
    file: str = typer.Option(..., help="Path to a single PDF or image."),
    question: str = typer.Option(..., help="Question to ask about that document."),
    generate_k: int = typer.Option(3, help="Pages sent to Qwen2.5-VL."),
):
    """
    Answers a question about ONE document, in memory, no database.
    """

    answer = phase2.ask_document(
        file=file, question=question, generate_k=generate_k,
    )
    _print(question, answer)


@app.command()
def db(
    question: str = typer.Option(..., help="Question to ask against the database."),
    retrieve_k: int = typer.Option(10, help="Pages pulled from Qdrant."),
    generate_k: int = typer.Option(3, help="Pages sent to Qwen2.5-VL."),
):
    """
    Answers a question using the whole indexed Qdrant database.
    """

    answer = phase2.ask_database(
        question=question, retrieve_k=retrieve_k, generate_k=generate_k,
    )
    _print(question, answer)


def _print(question, answer):
    """
    Pretty-prints the question, answer and supporting sources.

    Args:
        question (str): The original user question.
        answer (phase2.Answer): The pipeline output.
    """

    sep = "=" * 70
    console.print(f"\n[bold]{sep}[/]")
    console.print(f"[bold cyan]Question[/] : {question}")
    console.print(f"[bold]{sep}[/]\n")
    console.print(f"[bold green]Answer[/] :\n{answer.text}\n")

    if answer.cited:
        console.print(f"[bold]Cited pages[/] ({len(answer.cited)}) :")
        for i, src in enumerate(answer.cited, 1):
            console.print(
                f"  [{i}] [dim]{src.document} p.{src.page} score={src.score:.3f}[/]"
            )

    if answer.sources:
        console.print(f"\n[bold]Retrieved[/] ({len(answer.sources)}) :")
        for i, src in enumerate(answer.sources, 1):
            console.print(
                f"  [{i}] [dim]{src.document} p.{src.page} score={src.score:.3f}[/]"
            )


if __name__ == "__main__":
    app()
