"""
Ollama client — text and vision generation for both pipelines.

Pipeline 1 sends a text prompt to a small text model (Mistral 7B by
default). Pipeline 2 sends a prompt + page images to a vision model
(Qwen2.5-VL 7B by default) and asks for a strict JSON envelope so we
can extract cited pages reliably.

Requires :  pip install ollama  (also installs httpx).
"""

import base64
import io
import json
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


### Public API ###

def generate_text(prompt, *, model, host, keep_alive="5m"):
    """
    Calls Ollama text generation with a single user prompt.

    Args:
        prompt (str): The full prompt sent to the model.
        model (str): Ollama model tag, e.g. "mistral:7b-instruct".
        host (str): Ollama daemon URL.
        keep_alive (str): How long to keep the model warm.

    Returns:
        str: The model's text completion, stripped of whitespace.
    """

    client = _client(host)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        keep_alive=keep_alive,
        options={"temperature": 0.0},
    )
    return response["message"]["content"].strip()


@dataclass(frozen=True)
class VLMAnswer:
    """
    Structured answer from the vision model.

    Attributes:
        answer: The natural-language answer.
        cited_pages: Indices (1-based) of the input pages the model cited.
        raw: The raw JSON content returned by the model.
    """

    answer: str
    cited_pages: list
    raw: str


def generate_vision(prompt, images, *, model, host, keep_alive="5m"):
    """
    Calls Ollama vision generation with images attached.

    The system prompt forces a JSON envelope with `answer` and
    `cited_pages` so the caller can robustly extract the answer and
    the indices of the pages the model actually used.

    Args:
        prompt (str): The user question.
        images (list[PIL.Image.Image]): The page images, in retrieval order.
        model (str): Ollama vision model tag.
        host (str): Ollama daemon URL.
        keep_alive (str): How long to keep the model warm.

    Returns:
        VLMAnswer: Structured answer with extracted cited pages.
    """

    client = _client(host)
    encoded = [_image_to_b64(img) for img in images]

    system = (
        "You are a careful document assistant. The user provides one or "
        "more page images, numbered 1..N in the order they appear. "
        "Reply with ONE JSON object only, with this exact shape :\n"
        '{"answer": "...", "cited_pages": [1, 2]}\n'
        "Use cited_pages to list the 1-based indices of the pages that "
        "support your answer. If no page contains the answer, return "
        '{"answer": "Information non trouvée dans les documents fournis.", '
        '"cited_pages": []}.'
    )

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt, "images": encoded},
        ],
        keep_alive=keep_alive,
        format="json",
        options={"temperature": 0.0},
    )

    raw = response["message"]["content"].strip()
    return _parse_vlm_response(raw, n_pages=len(images))


### Helpers ###

def _client(host):
    """
    Returns an Ollama client bound to the given host.

    Args:
        host (str): Ollama daemon URL.

    Returns:
        ollama.Client: Ready-to-use synchronous client.
    """

    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "The ollama package is required.  Install with :  pip install ollama"
        ) from exc

    return ollama.Client(host=host)


def _image_to_b64(image):
    """
    Encodes a PIL image as base64 PNG, the format Ollama expects.

    Args:
        image (PIL.Image.Image): The page image.

    Returns:
        str: Base64-encoded PNG bytes.
    """

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_vlm_response(raw, *, n_pages):
    """
    Parses Ollama's JSON envelope, defensively.

    Args:
        raw (str): The raw content string returned by the model.
        n_pages (int): Number of input pages, for citation validation.

    Returns:
        VLMAnswer: Validated answer with cited_pages in [1, n_pages].
    """

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("VLM did not return valid JSON, treating raw as answer.")
        return VLMAnswer(answer=raw, cited_pages=[], raw=raw)

    answer = str(payload.get("answer", "")).strip()
    cited = payload.get("cited_pages", [])

    # Defensively coerce to integers in 1..n_pages, dedup, preserve order.
    valid = []
    seen = set()
    for item in cited if isinstance(cited, list) else []:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n_pages and idx not in seen:
            valid.append(idx)
            seen.add(idx)

    return VLMAnswer(answer=answer, cited_pages=valid, raw=raw)
