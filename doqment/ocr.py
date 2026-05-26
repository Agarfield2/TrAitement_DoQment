"""
Tesseract OCR — fallback for Pipeline 1 when no SROIE annotation exists.

The canonical OCREngine class in ingestion.py has a known bug (its
_load() method references a non-existent self._model attribute). We
can't patch ingestion.py — it is a canonical artefact. Instead, we
expose a Tesseract wrapper that produces ingestion.TextLine instances
directly consumable by ingestion.group_lines_into_passages().

Pre-processing mirrors what the teammates' pdf_ocr.py uses : grayscale
→ adaptive Gaussian threshold → light dilate. The pytesseract output
is grouped from word-level back to line-level via the block/par/line
identifiers so the chunker sees lines, not isolated words.
"""

import logging


logger = logging.getLogger(__name__)


### Public API ###

def ocr_image(image, lang="fra+eng", preprocess=True):
    """
    Runs Tesseract OCR on a PIL image and returns ingestion.TextLine list.

    Args:
        image (PIL.Image.Image): The image to OCR.
        lang (str): Tesseract language code(s), e.g. "fra+eng".
            Falls back automatically if the requested code is missing.
        preprocess (bool): If True, run adaptive thresholding first.

    Returns:
        list: ingestion.TextLine instances, one per detected line.
    """

    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "Tesseract OCR fallback requires pytesseract. "
            "Install with :  pip install pytesseract"
        ) from exc

    # Best-effort explicit location. If we find a binary, point
    # pytesseract at it. Otherwise let pytesseract try its own default
    # ("tesseract" via subprocess) — sometimes that resolves the binary
    # in environments where our shutil.which probe does not.
    binary = _find_tesseract_binary()
    if binary is not None:
        pytesseract.pytesseract.tesseract_cmd = binary

    chosen_lang = _pick_lang(pytesseract, lang)
    img = _preprocess(image) if preprocess else image.convert("RGB")

    try:
        data = pytesseract.image_to_data(
            img,
            lang=chosen_lang,
            config="--oem 3 --psm 3",
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "The `tesseract` binary could not be located. Install it "
            "(e.g. `sudo dnf install tesseract` or `sudo apt install "
            "tesseract-ocr tesseract-ocr-fra`). If it is installed but "
            "the launching shell has a minimal PATH (some desktop apps "
            "do this), set DOQMENT_TESSERACT_PATH to its absolute path "
            "(`/usr/bin/tesseract` on Fedora and Debian)."
        ) from exc

    return _group_into_lines(data)


def _find_tesseract_binary():
    """
    Locates the `tesseract` executable.

    Strategy : honour `DOQMENT_TESSERACT_PATH` if set ; otherwise try
    `shutil.which` on PATH ; otherwise fall back to a small list of
    common Unix install locations. This last step handles the case
    where Streamlit was launched from a desktop terminal (Obsidian,
    VS Code on some platforms, …) that ships with a minimal PATH not
    including /usr/bin or /usr/local/bin.

    Returns:
        str | None: Absolute path to tesseract, or None if not found.
    """

    import os
    import shutil

    override = os.environ.get("DOQMENT_TESSERACT_PATH")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override

    binary = shutil.which("tesseract")
    if binary:
        return binary

    fallbacks = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        "/opt/local/bin/tesseract",
    ]
    for path in fallbacks:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


### Helpers ###

def _preprocess(image):
    """
    Adaptive-threshold preprocessing borrowed from the teammates' pdf_ocr.py.

    Args:
        image (PIL.Image.Image): The raw input image.

    Returns:
        PIL.Image.Image: A binarised image ready for Tesseract.
    """

    try:
        import cv2
    except ImportError:
        return image.convert("L")

    import numpy as np
    from PIL import Image

    arr = np.array(image.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    return Image.fromarray(binary)


def _pick_lang(pytesseract, requested):
    """
    Returns a lang code that Tesseract can actually serve.

    Args:
        pytesseract: The pytesseract module.
        requested (str): The user-requested lang code.

    Returns:
        str: A lang code that is installed locally.
    """

    try:
        available = pytesseract.get_languages()
    except Exception:
        return requested

    if not available:
        return requested

    parts = [p for p in requested.split("+") if p in available]
    if parts:
        return "+".join(parts)
    for fallback in ("fra", "fr", "eng", "en"):
        if fallback in available:
            logger.warning(
                "Tesseract lang '%s' missing, falling back to '%s'",
                requested, fallback,
            )
            return fallback
    return available[0]


def _group_into_lines(data):
    """
    Groups word-level Tesseract output into line-level TextLines.

    Args:
        data (dict): The dict returned by pytesseract.image_to_data
            with output_type=Output.DICT.

    Returns:
        list: ingestion.TextLine instances with axis-aligned bboxes.
    """

    from ingestion import BoundingBox, TextLine

    by_line = {}
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        by_line.setdefault(key, []).append(i)

    lines = []
    for key in sorted(by_line.keys()):
        idxs = by_line[key]
        words = [data["text"][i].strip() for i in idxs]
        text = " ".join(w for w in words if w)
        if not text:
            continue

        xs = [int(data["left"][i]) for i in idxs]
        ys = [int(data["top"][i]) for i in idxs]
        rs = [int(data["left"][i]) + int(data["width"][i]) for i in idxs]
        bs = [int(data["top"][i]) + int(data["height"][i]) for i in idxs]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(rs), max(bs)

        confs = []
        for i in idxs:
            try:
                c = float(data["conf"][i])
                if c >= 0:
                    confs.append(c / 100.0)
            except (TypeError, ValueError):
                pass
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        bbox = BoundingBox(
            x1=float(x_min), y1=float(y_min),
            x2=float(x_max), y2=float(y_min),
            x3=float(x_max), y3=float(y_max),
            x4=float(x_min), y4=float(y_max),
        )
        lines.append(TextLine(text=text, bbox=bbox, confidence=avg_conf))

    return lines
