"""
Evaluation script - Tesseract / docTR (/ Surya / Kraken) vs SROIE ground-truth.

USAGE
  # Full dataset (all engines) :
  python Comparaison_OCR.py --images data/SROIE2019/task1&2_test(361p) \\
                             --refs   "data/SROIE2019/text.task1&2-test（361p)"

  # Quick sample (10 files) :
  python Comparaison_OCR.py --images data/SROIE2019/task1&2_test(361p) \\
                             --refs   "data/SROIE2019/text.task1&2-test（361p)" \\
                             --max-docs 10

  # Select specific engines :
  python Comparaison_OCR.py ... --engines tesseract doctr

  # Without contrast enhancement (Tesseract) :
  python Comparaison_OCR.py ... --enhance off

  # Custom contrast parameters (Tesseract) :
  python Comparaison_OCR.py ... --alpha 2.0 --beta 10

  # Grid search - test all enhance/alpha/beta combos on Tesseract :
  python Comparaison_OCR.py --engines tesseract --tesseract-grid --max-docs 20

Engines available :
  tesseract  - classic OCR, requires pytesseract + Tesseract binary
  doctr      - deep-learning OCR by Mindee (pip install python-doctr)
  surya      - transformer-based multilingual OCR (pip install surya-ocr)
  kraken     - sequence-to-sequence OCR, strong on historical docs (pip install kraken)

Metrics reported per document and globally (per engine) :
  CER (Character Error Rate), WER (Word Error Rate),
  Precision / Recall / F1, SequenceMatcher similarity.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

import fitz
import numpy as np
from PIL import Image
import pytesseract

# Tesseract path : Windows default, overridable via env var
import platform as _platform
_default_tesseract = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if _platform.system() == "Windows"
    else "/usr/bin/tesseract"
)
pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "DOQMENT_TESSERACT_PATH", _default_tesseract
)


# ── Data structures ──────────────────────────────────────────────────────────

class TextLine:
    def __init__(self, text, bbox=None, confidence=1.0):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence


# ── Image loading ─────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".png", ".jpeg", ".tif", ".tiff", ".pdf"}


def pdf_to_image(pdf_path: Path, dpi: int = 300) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def load_image(path: Path, dpi: int = 300) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        return pdf_to_image(path, dpi)
    return Image.open(path).convert("RGB")


# ── Preprocessing ─────────────────────────────────────────────────────────────

# Images whose grayscale RMS std-dev is below this threshold are considered
# low-contrast and will have the contrast filter applied.
# Calibrated on 360 SROIE receipts: threshold=20 restricts boosting to
# near-illegible docs (RMS < 20). Raise to apply more aggressively.
LOW_CONTRAST_RMS = 20.0


def needs_contrast_boost(img: Image.Image,
                         threshold: float = LOW_CONTRAST_RMS) -> bool:
    """
    Returns True if the image is low-contrast and would benefit from
    contrast enhancement.

    Uses the std-dev of grayscale pixel intensities as a proxy for contrast.
    A low std-dev means a narrow pixel range (faded/washed-out document).
    A high std-dev means strong black/white separation - boosting would
    over-saturate and degrade Tesseract accuracy.
    """
    arr = np.array(img.convert("L"), dtype=float)
    return float(arr.std()) < threshold


def enhance_contrast(img: Image.Image, alpha: float = 1.5,
                     beta: int = 0) -> Image.Image:
    """
    Linear contrast boost via cv2.convertScaleAbs.

    Call needs_contrast_boost() first to decide whether to apply this.

    Args:
        alpha: Contrast multiplier (1.0 = unchanged, >1 = more contrast).
        beta:  Brightness offset in pixel units.
    """
    try:
        import cv2
    except (ImportError, AttributeError):
        return img

    img_np = np.array(img.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    enhanced = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced)


def preprocess(img: Image.Image, enhance="auto",
               alpha: float = 1.5, beta: int = 0) -> Image.Image:
    """
    Full pipeline : contrast (adaptive) → adaptive threshold → dilate.

    enhance="auto"  - apply contrast filter only if needs_contrast_boost().
    enhance=True    - always apply.
    enhance=False   - never apply.
    """
    try:
        import cv2
    except (ImportError, AttributeError):
        return img.convert("L")

    if enhance == "auto":
        do_enhance = needs_contrast_boost(img)
    else:
        do_enhance = bool(enhance)

    if do_enhance:
        img = enhance_contrast(img, alpha=alpha, beta=beta)

    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return Image.fromarray(cv2.dilate(binary, kernel, iterations=1))


# ── OCR (Tesseract only) ──────────────────────────────────────────────────────

def ocr_tesseract_lines(img: Image.Image, lang: str,
                        enhance="auto",
                        alpha: float = 1.5,
                        beta: int = 0) -> list:
    processed = preprocess(img, enhance=enhance, alpha=alpha, beta=beta)
    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        config="--oem 3 --psm 3",
        output_type=pytesseract.Output.DICT,
    )
    lines = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        conf_raw = data["conf"][i]
        conf = float(conf_raw) / 100.0 if conf_raw != "-1" else 0.0
        x, y, w, h = (data["left"][i], data["top"][i],
                      data["width"][i], data["height"][i])
        lines.append(TextLine(text=text, bbox=(x, y, x + w, y + h),
                              confidence=conf))
    return lines


def lines_to_text(lines: list) -> str:
    return " ".join(l.text for l in lines)


# ── OCR - docTR ───────────────────────────────────────────────────────────────

def ocr_doctr_lines(img: Image.Image) -> list:
    """
    Run docTR on a PIL image.
    Install : pip install python-doctr[torch]   (or [tf] for TensorFlow)
    docTR returns a Document → Pages → Blocks → Lines → Words hierarchy.
    """
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError:
        raise ImportError(
            "docTR not installed. Run: pip install python-doctr[torch]"
        )

    # Lazy-load model (cached after first call)
    if not hasattr(ocr_doctr_lines, "_model"):
        ocr_doctr_lines._model = ocr_predictor(pretrained=True)

    import tempfile, io
    # docTR works best from file; write image to a temp PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        doc = DocumentFile.from_images(tmp.name)

    result = ocr_doctr_lines._model(doc)
    lines = []
    for page in result.pages:
        h, w = page.dimensions
        for block in page.blocks:
            for line in block.lines:
                words = [word.value for word in line.words]
                text = " ".join(words).strip()
                if not text:
                    continue
                # Aggregate bounding box from all words (relative → pixels)
                xs = [word.geometry[0][0] * w for word in line.words] + \
                     [word.geometry[1][0] * w for word in line.words]
                ys = [word.geometry[0][1] * h for word in line.words] + \
                     [word.geometry[1][1] * h for word in line.words]
                bbox = (min(xs), min(ys), max(xs), max(ys))
                conf = float(np.mean([word.confidence for word in line.words]))
                lines.append(TextLine(text=text, bbox=bbox, confidence=conf))
    return lines


# ── OCR - Surya ───────────────────────────────────────────────────────────────

def ocr_surya_lines(img: Image.Image, langs: list = None) -> list:
    """
    Run Surya OCR on a PIL image.
    Install : pip install surya-ocr
    langs   : list of BCP-47 language codes, e.g. ["fr", "en"].
              Defaults to ["fr", "en"] when None.

    Supports surya-ocr >= 0.20 (new API) and older versions.
    """
    if langs is None:
        langs = ["fr", "en"]

    # ── surya >= 0.20 : SuryaInferenceManager API ────────────────────────────
    try:
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        if not hasattr(ocr_surya_lines, "_models_v2"):
            manager = SuryaInferenceManager()
            ocr_surya_lines._models_v2 = RecognitionPredictor(manager)
        rec_predictor = ocr_surya_lines._models_v2

        results = rec_predictor([img])
        lines = []
        for page_result in results:
            # v0.20 returns .blocks with .html and .bbox
            blocks = getattr(page_result, "blocks", None)
            if blocks is not None:
                from bs4 import BeautifulSoup
                for block in blocks:
                    soup = BeautifulSoup(getattr(block, "html", ""), "html.parser")
                    text = soup.get_text(" ", strip=True)
                    if not text:
                        continue
                    bbox = tuple(block.bbox) if hasattr(block, "bbox") else None
                    conf = float(getattr(block, "confidence", 1.0))
                    lines.append(TextLine(text=text, bbox=bbox, confidence=conf))
            else:
                # fallback: .text_lines (older 0.20.x builds)
                for line in getattr(page_result, "text_lines", []):
                    text = line.text.strip()
                    if not text:
                        continue
                    bbox = tuple(line.bbox)
                    conf = float(line.confidence)
                    lines.append(TextLine(text=text, bbox=bbox, confidence=conf))
        return lines

    except ImportError:
        pass  # fall through to intermediate API

    # ── surya ~0.10-0.19 : RecognitionPredictor(det_predictor=...) ────────────
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        if not hasattr(ocr_surya_lines, "_models_mid"):
            ocr_surya_lines._models_mid = (
                RecognitionPredictor(),
                DetectionPredictor(),
            )
        rec_predictor, det_predictor = ocr_surya_lines._models_mid
        results = rec_predictor([img], [langs], det_predictor=det_predictor)
        lines = []
        for page_result in results:
            for line in page_result.text_lines:
                text = line.text.strip()
                if not text:
                    continue
                bbox = tuple(line.bbox)
                conf = float(line.confidence)
                lines.append(TextLine(text=text, bbox=bbox, confidence=conf))
        return lines

    except (ImportError, TypeError):
        pass  # fall through to legacy API

    # ── surya < 0.10 : run_ocr() ──────────────────────────────────────────────
    try:
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.processor import load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
    except ImportError:
        raise ImportError("Surya not installed. Run: pip install surya-ocr")

    if not hasattr(ocr_surya_lines, "_models_legacy"):
        ocr_surya_lines._models_legacy = (
            load_det_model(), load_det_processor(),
            load_rec_model(), load_rec_processor(),
        )
    det_model, det_processor, rec_model, rec_processor = ocr_surya_lines._models_legacy
    results = run_ocr([img], [langs], det_model, det_processor,
                      rec_model, rec_processor)
    lines = []
    for page_result in results:
        for line in page_result.text_lines:
            text = line.text.strip()
            if not text:
                continue
            bbox = tuple(line.bbox)
            conf = float(line.confidence)
            lines.append(TextLine(text=text, bbox=bbox, confidence=conf))
    return lines


# ── OCR - Kraken ──────────────────────────────────────────────────────────────

def ocr_kraken_lines(img: Image.Image, model_path: str = None) -> list:
    """
    Run Kraken OCR on a PIL image.
    Install : pip install kraken
    model_path : path to a .mlmodel file.
                 Defaults to the bundled English model if None.

    Kraken pipeline : binarize → segment → recognize.

    Windows note : Kraken reads text files with the system locale encoding
    which on French/English Windows is cp1252 and crashes on certain model
    metadata bytes.  We temporarily patch builtins.open to force UTF-8 for
    the duration of model loading.
    """
    import builtins

    _real_open = builtins.open

    def _utf8_open(file, mode="r", buffering=-1, encoding=None,
                   errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding is None:
            encoding = "utf-8"
            errors = errors or "replace"
        return _real_open(file, mode, buffering, encoding,
                          errors, newline, closefd)

    try:
        from kraken import binarization, pageseg, rpred
        from kraken.lib import models as kraken_models
    except ImportError:
        raise ImportError(
            "Kraken not installed. Run: pip install kraken"
        )

    # Lazy-load model (cached per model_path key)
    cache_key = model_path or "__default__"
    if not hasattr(ocr_kraken_lines, "_models"):
        ocr_kraken_lines._models = {}
    if cache_key not in ocr_kraken_lines._models:
        builtins.open = _utf8_open          # patch before loading
        try:
            if model_path:
                ocr_kraken_lines._models[cache_key] = kraken_models.load_any(model_path)
            else:
                # kraken.repo.get_model was removed in kraken >= 4.x.
                # The model must be downloaded once via CLI:
                #   kraken get 10.5281/zenodo.2577813
                # then passed with --kraken-model <path>.
                # We try a few common default locations before giving up.
                import pathlib
                # Search in ~/.kraken and in htrmopo cache (kraken >= 4.x)
                # htrmopo stores models in:
                #   Windows : %LOCALAPPDATA%/htrmopo/htrmopo/<uuid>/
                #   Linux   : ~/.local/share/htrmopo/<uuid>/
                import os as _os
                local_app = pathlib.Path(
                    _os.environ.get("LOCALAPPDATA",
                                    pathlib.Path.home() / ".local" / "share")
                )
                htrmopo_base = local_app / "htrmopo" / "htrmopo"
                mlmodels_htrmopo = sorted(htrmopo_base.rglob("*.mlmodel")) if htrmopo_base.exists() else []
                candidates = [
                    pathlib.Path.home() / ".kraken" / "en_best.mlmodel",
                    pathlib.Path.home() / ".kraken" / "en-default.mlmodel",
                ] + mlmodels_htrmopo
                found = next((p for p in candidates if p.exists()), None)
                if found:
                    ocr_kraken_lines._models[cache_key] = kraken_models.load_any(str(found))
                else:
                    raise FileNotFoundError(
                        "No default Kraken model found. Download one first:\n"
                        "  kraken get 10.5281/zenodo.2577813\n"
                        "Then pass it explicitly:\n"
                        "  --kraken-model ~/.kraken/<model>.mlmodel"
                    )
        finally:
            builtins.open = _real_open      # always restore

    model = ocr_kraken_lines._models[cache_key]

    # Kraken expects a grayscale/binarized PIL image
    bw = binarization.nlbin(img.convert("L"))

    # kraken >= 4.x removed return_lines; segment() now always returns a
    # RecognitionResults-compatible object with a .lines attribute.
    import inspect as _inspect
    seg_kwargs = {"text_direction": "horizontal-lr"}
    if "return_lines" in _inspect.signature(pageseg.segment).parameters:
        seg_kwargs["return_lines"] = True   # kraken < 4.x compat
    seg = pageseg.segment(bw, **seg_kwargs)

    pred = rpred.rpred(model, bw, seg)

    lines = []
    for record in pred:
        text = record.prediction.strip()
        if not text:
            continue
        # record.line.boundary is a list of (x, y) polygon points
        if hasattr(record, "line") and hasattr(record.line, "boundary") and record.line.boundary:
            pts = record.line.boundary
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            bbox = None
        conf = float(np.mean(record.confidences)) if getattr(record, "confidences", None) else 0.0
        lines.append(TextLine(text=text, bbox=bbox, confidence=conf))
    return lines


# ── Reference parser (SROIE format) ──────────────────────────────────────────

def parse_ref_file(ref_path: Path) -> str:
    lines = []
    with open(ref_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            text = ",".join(parts[8:]) if len(parts) >= 9 else line
            if text:
                lines.append(text)
    return "\n".join(lines)


# ── Metrics ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def edit_distance(a, b) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len(b) + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] \
                    else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[-1]


def cer(ref, hyp) -> float:
    ref, hyp = normalize(ref), normalize(hyp)
    return 0.0 if not ref else min(edit_distance(ref, hyp) / len(ref), 1.0)


def wer(ref, hyp) -> float:
    ref_w, hyp_w = normalize(ref).split(), normalize(hyp).split()
    return 0.0 if not ref_w \
           else min(edit_distance(ref_w, hyp_w) / len(ref_w), 1.0)


def prf(ref, hyp):
    ref_w = set(normalize(ref).split())
    hyp_w = set(normalize(hyp).split())
    if not hyp_w:
        return 0.0, 0.0, 0.0
    tp = len(ref_w & hyp_w)
    p = tp / len(hyp_w)
    r = tp / len(ref_w) if ref_w else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def similarity(ref, hyp) -> float:
    return SequenceMatcher(None, normalize(ref), normalize(hyp)).ratio()


# ── File pairing ──────────────────────────────────────────────────────────────

def find_pairs(images_dir: Path, refs_dir: Path) -> list:
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        ref = refs_dir / (img.stem + ".txt")
        if ref.exists():
            pairs.append((img, ref))
    return pairs


# ── Reporting ─────────────────────────────────────────────────────────────────

def fmt(x: float) -> str:
    return f"{x * 100:6.2f}%"


def print_summary(results_by_engine: dict) -> None:
    for engine, results in results_by_engine.items():
        n = len(results)
        if not n:
            continue
        avg = lambda k: sum(r[k] for r in results) / n
        print("\n" + "=" * 60)
        print(f"GLOBAL SUMMARY - {engine.upper()}")
        print("=" * 60)
        print(f"Files : {n}")
        print(f"CER   : {fmt(avg('cer'))}")
        print(f"WER   : {fmt(avg('wer'))}")
        print(f"Prec  : {fmt(avg('p'))}")
        print(f"Recall: {fmt(avg('r'))}")
        print(f"F1    : {fmt(avg('f1'))}")
        print(f"Sim   : {fmt(avg('sim'))}")


def print_grid_summary(grid_results: dict) -> None:
    """
    Print a ranked table of all Tesseract preprocessing combos.
    grid_results : { (enhance, alpha, beta): [metric_dicts] }
    """
    rows = []
    for (enhance, alpha, beta), results in grid_results.items():
        n = len(results)
        if not n:
            continue
        avg = lambda k: sum(r[k] for r in results) / n
        rows.append({
            "enhance": enhance, "alpha": alpha, "beta": beta,
            "cer": avg("cer"), "wer": avg("wer"),
            "f1":  avg("f1"),  "sim": avg("sim"), "n": n,
        })

    rows.sort(key=lambda r: r["f1"], reverse=True)

    print("\n" + "=" * 75)
    print("TESSERACT GRID SEARCH - ranked by F1")
    print("=" * 75)
    print(f"{'enhance':<8} {'alpha':>6} {'beta':>5}  {'CER':>7} {'WER':>7} {'F1':>7} {'Sim':>7}  n")
    print("-" * 75)
    for r in rows:
        print(f"{r['enhance']:<8} {r['alpha']:>6.2f} {r['beta']:>5d}  "
              f"{fmt(r['cer'])} {fmt(r['wer'])} {fmt(r['f1'])} {fmt(r['sim'])}  {r['n']}")
    print("=" * 75)
    if rows:
        best = rows[0]
        print(f"\n★  Best combo : --enhance {best['enhance']} --alpha {best['alpha']} --beta {best['beta']}"
              f"  →  F1={fmt(best['f1'])}  CER={fmt(best['cer'])}")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_ENGINES = ["tesseract", "doctr", "surya", "kraken"]

# Grid search parameter space for Tesseract preprocessing
GRID_ENHANCE = ["off", "auto", "on"]
GRID_ALPHA   = [1.0, 1.5, 2.0, 2.5]
GRID_BETA    = [0, 10, 20]


def run_engine(engine: str, img: Image.Image, args,
               enhance=None, alpha=None, beta=None) -> list:
    """Dispatch image to the right OCR engine and return TextLine list."""
    if engine == "tesseract":
        enh   = enhance if enhance is not None else {"auto": "auto", "on": True, "off": False}[args.enhance]
        alph  = alpha   if alpha   is not None else args.alpha
        bet   = beta    if beta    is not None else args.beta
        return ocr_tesseract_lines(img, args.lang,
                                   enhance=enh, alpha=alph, beta=bet)
    elif engine == "doctr":
        return ocr_doctr_lines(img)
    elif engine == "surya":
        langs = [l.strip() for l in args.lang.split("+")]
        return ocr_surya_lines(img, langs=langs)
    elif engine == "kraken":
        return ocr_kraken_lines(img, model_path=args.kraken_model)
    else:
        raise ValueError(f"Unknown engine: {engine}")


def main():
    parser = argparse.ArgumentParser(
        description="OCR evaluation (Tesseract / docTR / Surya / Kraken) "
                    "against SROIE ground-truth"
    )
    parser.add_argument("--images",
                        default="data/SROIE2019/task1&2_test(361p)")
    parser.add_argument("--refs",
                        default="data/SROIE2019/text.task1&2-test（361p)")
    parser.add_argument("--engines", nargs="+", default=ALL_ENGINES,
                        choices=ALL_ENGINES,
                        help="Engines to run (default: all four)")
    parser.add_argument("--lang",     default="fra+eng",
                        help="Language(s) for Tesseract (e.g. fra+eng) and "
                             "Surya (BCP-47 codes joined by '+', e.g. fr+en)")
    parser.add_argument("--dpi",      type=int, default=300)
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit processing to N files")
    parser.add_argument("--enhance",  default="auto",
                        choices=["auto", "on", "off"],
                        help="Contrast filter for Tesseract: auto (default) = "
                             "apply only on low-contrast images, on = always, "
                             "off = never")
    parser.add_argument("--alpha",    type=float, default=1.5,
                        help="Contrast multiplier for Tesseract (default 1.5)")
    parser.add_argument("--beta",     type=int,   default=0,
                        help="Brightness offset for Tesseract (default 0)")
    parser.add_argument("--kraken-model", default=None,
                        help="Path to a Kraken .mlmodel file "
                             "(default: downloads en_best.mlmodel)")
    parser.add_argument("--tesseract-grid", action="store_true",
                        help="Grid search over Tesseract preprocessing params "
                             "(enhance x alpha x beta). Ignores --enhance/--alpha/--beta.")
    args = parser.parse_args()

    images_dir = Path(args.images)
    refs_dir   = Path(args.refs)

    pairs = find_pairs(images_dir, refs_dir)
    if not pairs:
        print("No image/ref pairs found.")
        sys.exit(1)

    if args.max_docs is not None:
        pairs = pairs[:args.max_docs]
        print(f"Sample limited to {len(pairs)} file(s) "
              f"(--max-docs {args.max_docs})")

    # ── Grid search mode ──────────────────────────────────────────────────────
    if args.tesseract_grid:
        combos = [
            (enh, alph, bet)
            for enh  in GRID_ENHANCE
            for alph in GRID_ALPHA
            for bet  in GRID_BETA
        ]
        total = len(combos) * len(pairs)
        print(f"Tesseract grid search : {len(combos)} combos × {len(pairs)} files = {total} runs")

        grid_results = {c: [] for c in combos}

        for img_path, ref_path in pairs:
            print(f"\n── {img_path.name} ──")
            try:
                img = load_image(img_path, args.dpi)
            except Exception as exc:
                print(f"  [WARN] {exc}")
                continue
            ref = parse_ref_file(ref_path)

            for (enh, alph, bet) in combos:
                enh_mode = {"auto": "auto", "on": True, "off": False}[enh]
                label = f"enhance={enh} alpha={alph} beta={bet}"
                try:
                    lines = ocr_tesseract_lines(img, args.lang,
                                                enhance=enh_mode,
                                                alpha=alph, beta=bet)
                except Exception as exc:
                    print(f"  [{label}] ERROR - {exc}")
                    continue

                hyp      = lines_to_text(lines)
                c        = cer(ref, hyp)
                w        = wer(ref, hyp)
                p, r, f1 = prf(ref, hyp)
                s        = similarity(ref, hyp)
                grid_results[(enh, alph, bet)].append(
                    {"cer": c, "wer": w, "p": p, "r": r, "f1": f1, "sim": s}
                )
                print(f"  [{label}]  CER={fmt(c)} F1={fmt(f1)}")

        print_grid_summary(grid_results)
        return

    # ── Normal mode ───────────────────────────────────────────────────────────
    print(f"Engines : {', '.join(args.engines)}")
    results_by_engine = {e: [] for e in args.engines}

    for img_path, ref_path in pairs:
        print(f"\n── {img_path.name} ──")
        try:
            img = load_image(img_path, args.dpi)
        except Exception as exc:
            print(f"  [WARN] Could not load image: {exc}")
            continue

        ref = parse_ref_file(ref_path)

        for engine in args.engines:
            print(f"  [{engine}]", end="  ", flush=True)
            try:
                lines = run_engine(engine, img, args)
            except ImportError as exc:
                print(f"SKIP - {exc}")
                continue
            except Exception as exc:
                print(f"ERROR - {exc}")
                continue

            hyp      = lines_to_text(lines)
            c        = cer(ref, hyp)
            w        = wer(ref, hyp)
            p, r, f1 = prf(ref, hyp)
            s        = similarity(ref, hyp)

            results_by_engine[engine].append(
                {"cer": c, "wer": w, "p": p, "r": r, "f1": f1, "sim": s}
            )
            print(f"CER={fmt(c)} WER={fmt(w)} F1={fmt(f1)}")

    print_summary(results_by_engine)


if __name__ == "__main__":
    main()
