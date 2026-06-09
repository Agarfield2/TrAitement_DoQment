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


def generate_vision(prompt, images, *, model, host, keep_alive="5m",
                    num_ctx=8192, image_max_side=1024):
    """
    Calls Ollama vision generation with images attached and returns the
    model's plain-text answer.

    No JSON envelope is forced on the model : citations are handled by the
    caller (the retrieved pages sent in are the evidence). Forcing a JSON
    grammar made the 7B VLM degenerate under multi-image prompts, so we let
    it answer in natural language instead.

    Args:
        prompt (str): The user question.
        images (list[PIL.Image.Image]): The page images, in retrieval order.
        model (str): Ollama vision model tag.
        host (str): Ollama daemon URL.
        keep_alive (str): How long to keep the model warm.
        num_ctx (int): Ollama context window (token budget).
        image_max_side (int): Longest-side pixel cap per image.

    Returns:
        str: The model's answer text (empty string if it returned nothing).
    """

    client = _client(host)
    encoded = [_image_to_b64(img, max_side=image_max_side) for img in images]

    system = (
        "You are a careful document assistant. The user provides one or more "
        "page images. Answer the question using ONLY what is visible in those "
        "pages, concisely and in the user's language. If the answer is not "
        "present in the pages, say so plainly."
    )

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt, "images": encoded},
        ],
        keep_alive=keep_alive,
        options={
            "temperature": 0.0,
            # Fenêtre assez large pour contenir plusieurs images sans troncature
            # (la troncature corrompt le template et fait boucler le modèle).
            "num_ctx": num_ctx,
            # Garde-fous contre les boucles de répétition (<|im_start|>, addCriterion…).
            "repeat_penalty": 1.1,
            "num_predict": 1024,
        },
    )

    return response["message"]["content"].strip()


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


def _image_to_b64(image, max_side=None):
    """
    Encodes a PIL image as base64 PNG, the format Ollama expects.

    Args:
        image (PIL.Image.Image): The page image.
        max_side (int | None): If set, downscale so the longest side is at
            most this many pixels. Fewer pixels = far fewer vision tokens,
            which keeps multi-image prompts inside the model context.

    Returns:
        str: Base64-encoded PNG bytes.
    """

    image = image.convert("RGB")
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))   # preserves aspect ratio
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
