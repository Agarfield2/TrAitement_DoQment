"""
OCR fallback for Pipeline 1 when no SROIE annotation exists.

Two engines are exposed through ocr_image():
  - "doctr"     deep-learning recognizer (default), markedly more accurate
                on receipts ; needs python-doctr[torch].
  - "tesseract" classic binary path (the pre-processing pipeline below
                applies to this engine only).
The engine defaults to Settings.ocr_engine ("doctr", overridable via the
DOQMENT_OCR_ENGINE environment variable).

The canonical OCREngine class in ingestion.py has a known bug (its
_load() method references a non-existent self._model attribute). We
can't patch ingestion.py — it is a canonical artefact. Instead, we
expose a Tesseract wrapper that produces ingestion.TextLine instances
directly consumable by ingestion.group_lines_into_passages().

Pre-processing pipeline (applied in order when enabled):
  1. enhance_contrast  — cv2.convertScaleAbs (alpha/beta), only if the image
                         is detected as low-contrast (RMS < LOW_CONTRAST_RMS)
  2. adaptive threshold — grayscale → binary (Gaussian, 31×31, C=10)
  3. dilate             — 1×1 kernel, 1 pass

The contrast filter is applied adaptively: on well-contrasted documents it
is skipped to avoid over-saturation (which degrades Tesseract accuracy).
The threshold LOW_CONTRAST_RMS=20 was calibrated on 360 SROIE receipts:
raise it to apply the filter more aggressively (risks degrading normal scans),
lower it to restrict boosting to near-illegible documents only.
Pass enhance=True to force the filter regardless, enhance=False to disable it.
"""

import logging


logger = logging.getLogger(__name__)

# Images whose grayscale pixel RMS std-dev is below this threshold are
# considered low-contrast and will have the contrast filter applied.
#
# Calibrated on 360 SROIE receipts (evaluation run 2025-06):
#   - All docs in the task2train subset had RMS in the 24-41 range.
#   - Applying the filter on docs with RMS >= 20 degraded accuracy.
#   - Threshold of 20 restricts boosting to near-illegible documents
#     (severe fade, heavy shadow, scanner malfunction) while leaving
#     normal low-contrast receipts untouched.
LOW_CONTRAST_RMS = 20.0


### Public API ###

def ocr_image(image, engine=None, lang="fra+eng", preprocess=True,
              enhance="auto", alpha=1.5, beta=0):
    """
    Run OCR on a PIL image and return an ingestion.TextLine list.

    The engine is selected by `engine`, or — when None — by
    `doqment.settings.Settings.ocr_engine` (default "doctr", overridable
    via DOQMENT_OCR_ENGINE). Accepted values: "doctr" or "tesseract".

    The lang / preprocess / enhance / alpha / beta arguments only affect
    the Tesseract engine ; docTR uses its own pretrained pipeline.

    Returns:
        list: ingestion.TextLine instances, one per detected line.
    """

    if engine is None:
        from doqment.settings import load_settings
        engine = load_settings().ocr_engine
    engine = (engine or "doctr").lower()

    if engine == "doctr":
        return _ocr_doctr(image)
    if engine == "tesseract":
        return _ocr_tesseract(image, lang=lang, preprocess=preprocess,
                              enhance=enhance, alpha=alpha, beta=beta)
    raise ValueError(
        f"Unknown OCR engine {engine!r} (expected 'doctr' or 'tesseract')"
    )


def _ocr_tesseract(image, lang="fra+eng", preprocess=True,
                   enhance="auto", alpha=1.5, beta=0):
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
        preprocess (bool): If True, run the full preprocessing pipeline
            (contrast enhancement + adaptive threshold + dilate).
        enhance (bool | "auto"): Controls the contrast-boost step.
            "auto" (default) — apply only when the image RMS std-dev is
                below LOW_CONTRAST_RMS (adaptive, safe for all inputs).
            True  — always apply (useful to force on known faded scans).
            False — never apply (skip contrast step entirely).
        alpha (float): Contrast multiplier for enhance_contrast.
            1.0 = unchanged, >1.0 = more contrast. Default 1.5.
        beta (int): Brightness offset for enhance_contrast (+/- pixels).
            Default 0 (no brightness change).

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
    img = _preprocess(image, enhance=enhance, alpha=alpha, beta=beta) \
          if preprocess else image.convert("RGB")

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


### docTR engine ###

# Lazy-loaded singleton : the predictor is heavy and downloads weights on
# first use, so we build it once and reuse it across calls.
_DOCTR_MODEL = None


def _ocr_doctr(image):
    """
    Runs docTR on a PIL image and returns ingestion.TextLine list.

    docTR yields a Document -> Pages -> Blocks -> Lines -> Words hierarchy
    with geometry in relative coordinates ; we flatten it to line-level
    TextLines with axis-aligned pixel bounding boxes, matching the shape
    produced by the Tesseract path.

    Args:
        image (PIL.Image.Image): The image to OCR.

    Returns:
        list: ingestion.TextLine instances, one per detected line.
    """

    try:
        from doctr.models import ocr_predictor
    except ImportError as exc:
        raise ImportError(
            "docTR OCR engine requires python-doctr. "
            "Install with :  pip install 'python-doctr[torch]'"
        ) from exc

    import numpy as np
    from ingestion import BoundingBox, TextLine

    global _DOCTR_MODEL
    if _DOCTR_MODEL is None:
        logger.info("Loading docTR model (first call may download weights)...")
        _DOCTR_MODEL = ocr_predictor(pretrained=True)

    # docTR's predictor accepts a list of RGB numpy pages directly.
    page_np = np.array(image.convert("RGB"))
    result = _DOCTR_MODEL([page_np])

    lines = []
    for page in result.pages:
        h, w = page.dimensions
        for block in page.blocks:
            for line in block.lines:
                text = " ".join(word.value for word in line.words).strip()
                if not text:
                    continue

                xs, ys = [], []
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    xs.extend((x0 * w, x1 * w))
                    ys.extend((y0 * h, y1 * h))
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                bbox = BoundingBox(
                    x1=float(x_min), y1=float(y_min),
                    x2=float(x_max), y2=float(y_min),
                    x3=float(x_max), y3=float(y_max),
                    x4=float(x_min), y4=float(y_max),
                )
                conf = float(np.mean([word.confidence for word in line.words]))
                lines.append(TextLine(text=text, bbox=bbox, confidence=conf))

    return lines


### Helpers ###

def needs_contrast_boost(img, threshold=LOW_CONTRAST_RMS):
    """
    Returns True if the image is low-contrast and would benefit from
    contrast enhancement before OCR.

    Measures the standard deviation of pixel intensities on the grayscale
    image. A low std-dev means pixels are clustered around a narrow range
    (faded, washed-out, or uniformly dark) — the contrast filter helps.
    A high std-dev means the image already has strong black/white separation
    — the filter would over-saturate and degrade Tesseract accuracy.

    Args:
        img (PIL.Image.Image): Input image (any mode).
        threshold (float): RMS std-dev below which the filter is applied.
            Default LOW_CONTRAST_RMS = 45.0.

    Returns:
        bool: True if contrast enhancement is recommended.
    """
    import numpy as np
    arr = np.array(img.convert("L"), dtype=float)
    return float(arr.std()) < threshold


def enhance_contrast(img, alpha=1.5, beta=0):
    """
    Boosts contrast via cv2.convertScaleAbs (linear rescaling).

    Improves readability of faded receipts and low-contrast scans.
    Call needs_contrast_boost() first to decide whether to apply this.

    Args:
        img (PIL.Image.Image): Input RGB image.
        alpha (float): Contrast multiplier. 1.0 = unchanged, >1 = more.
        beta (int): Brightness offset in pixel value units.

    Returns:
        PIL.Image.Image: Contrast-enhanced RGB image.
    """

    try:
        import cv2
    except (ImportError, AttributeError):
        return img

    import numpy as np
    from PIL import Image

    img_np = np.array(img.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    enhanced = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced)


def _preprocess(image, enhance="auto", alpha=1.5, beta=0):
    """
    Full preprocessing pipeline: contrast (adaptive) → threshold → dilate.

    Args:
        image (PIL.Image.Image): Raw input image.
        enhance (bool | "auto"): See ocr_image() docstring.
        alpha (float): Contrast multiplier passed to enhance_contrast.
        beta (int): Brightness offset passed to enhance_contrast.

    Returns:
        PIL.Image.Image: Binarised image ready for Tesseract.
    """

    try:
        import cv2
    except (ImportError, AttributeError):
        return image.convert("L")

    import numpy as np
    from PIL import Image

    # Resolve "auto": apply only if the image looks low-contrast.
    if enhance == "auto":
        do_enhance = needs_contrast_boost(image)
        if do_enhance:
            logger.debug("Low-contrast image detected (RMS < %.1f), applying enhance_contrast.", LOW_CONTRAST_RMS)
        else:
            logger.debug("Image contrast sufficient (RMS >= %.1f), skipping enhance_contrast.", LOW_CONTRAST_RMS)
    else:
        do_enhance = bool(enhance)

    img = enhance_contrast(image, alpha=alpha, beta=beta) if do_enhance else image

    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return Image.fromarray(dilated)


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
