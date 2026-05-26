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

    The path to the `tesseract` binary is read from
    `doqment.settings.Settings.tesseract_cmd`, defaulting to
    `/usr/bin/tesseract`. Override either by editing settings.py or by
    setting the `DOQMENT_TESSERACT_PATH` environment variable. No
    auto-detection : the path is authoritative and explicit.

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

    from doqment.settings import load_settings
    pytesseract.pytesseract.tesseract_cmd = load_settings().tesseract_cmd

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
            f"Tesseract binary not reachable at "
            f"`{load_settings().tesseract_cmd}`. Either install it there "
            f"(e.g. `sudo dnf install tesseract`), edit "
            f"`doqment/settings.py` to point at the real location, or set "
            f"DOQMENT_TESSERACT_PATH=/path/to/tesseract before launching."
        ) from exc

    return _group_into_lines(data)


### Helpers ###

def _preprocess(image):
    """
    Adaptive-threshold preprocessing borrowed from the teammates' pdf_ocr.py.

    Args:
        image (PIL.Image.Image): The raw input image.

    Returns:
        PIL.Image.Image: A binarised image ready for Tesseract.
    """

    # cv2 can fail to import for several reasons : not installed at all,
    # ABI mismatch with the system NumPy (e.g. cv2 compiled against
    # NumPy 1 but NumPy 2 installed, which raises AttributeError instead
    # of ImportError), or a broken shared library. In every case we'd
    # rather skip preprocessing than fail the OCR entirely.
    try:
        import cv2
    except (ImportError, AttributeError):
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
