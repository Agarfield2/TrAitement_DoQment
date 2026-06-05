"""
Evaluation script — Tesseract OCR vs SROIE ground-truth.

USAGE
  # Full dataset :
  python Comparaison_OCR.py --images data/SROIE2019/task1&2_test(361p) \\
                             --refs   "data/SROIE2019/text.task1&2-test（361p)"

  # Quick sample (10 files) :
  python Comparaison_OCR.py --images data/SROIE2019/task1&2_test(361p) \\
                             --refs   "data/SROIE2019/text.task1&2-test（361p)" \\
                             --max-docs 10

  # Without contrast enhancement :
  python Comparaison_OCR.py ... --no-enhance

  # Custom contrast parameters :
  python Comparaison_OCR.py ... --alpha 2.0 --beta 10

Metrics reported per document and globally :
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

pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "DOQMENT_TESSERACT_PATH", "/usr/bin/tesseract"
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
    A high std-dev means strong black/white separation — boosting would
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

    enhance="auto"  — apply contrast filter only if needs_contrast_boost().
    enhance=True    — always apply.
    enhance=False   — never apply.
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


def print_summary(results: list) -> None:
    n = len(results)
    if not n:
        return
    avg = lambda k: sum(r[k] for r in results) / n
    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)
    print(f"Files : {n}")
    print(f"CER   : {fmt(avg('cer'))}")
    print(f"WER   : {fmt(avg('wer'))}")
    print(f"Prec  : {fmt(avg('p'))}")
    print(f"Recall: {fmt(avg('r'))}")
    print(f"F1    : {fmt(avg('f1'))}")
    print(f"Sim   : {fmt(avg('sim'))}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tesseract OCR evaluation against SROIE ground-truth"
    )
    parser.add_argument("--images",
                        default="data/SROIE2019/task1&2_test(361p)")
    parser.add_argument("--refs",
                        default="data/SROIE2019/text.task1&2-test（361p)")
    parser.add_argument("--lang",     default="fra+eng")
    parser.add_argument("--dpi",      type=int, default=300)
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit processing to N files")
    parser.add_argument("--enhance",  default="auto",
                        choices=["auto", "on", "off"],
                        help="Contrast filter: auto (default) = apply only on "
                             "low-contrast images, on = always, off = never")
    parser.add_argument("--alpha",    type=float, default=1.5,
                        help="Contrast multiplier (default 1.5)")
    parser.add_argument("--beta",     type=int,   default=0,
                        help="Brightness offset (default 0)")
    args = parser.parse_args()

    enhance_mode = {"auto": "auto", "on": True, "off": False}[args.enhance]

    images_dir = Path(args.images)
    refs_dir   = Path(args.refs)

    pairs = find_pairs(images_dir, refs_dir)
    if not pairs:
        print("No image/ref pairs found.")
        sys.exit(1)

    if args.max_docs is not None:
        pairs = pairs[:args.max_docs]
        print(f"Sample limited to {len(pairs)} file(s) (--max-docs {args.max_docs})")

    results = []
    for img_path, ref_path in pairs:
        print(f"Processing {img_path.name}", end="  ", flush=True)
        try:
            img  = load_image(img_path, args.dpi)
            lines = ocr_tesseract_lines(
                img, args.lang,
                enhance=enhance_mode,
                alpha=args.alpha,
                beta=args.beta,
            )
        except Exception as exc:
            print(f"[WARN] {exc}")
            continue

        hyp = lines_to_text(lines)
        ref = parse_ref_file(ref_path)

        c        = cer(ref, hyp)
        w        = wer(ref, hyp)
        p, r, f1 = prf(ref, hyp)
        s        = similarity(ref, hyp)

        results.append({"cer": c, "wer": w, "p": p, "r": r, "f1": f1, "sim": s})
        print(f"CER={fmt(c)} WER={fmt(w)}")

    print_summary(results)


if __name__ == "__main__":
    main()
