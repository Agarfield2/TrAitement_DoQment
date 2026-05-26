"""
  python evaluate_ocr.py --engine tesseract
  python evaluate_ocr.py --engine paddle
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from difflib import SequenceMatcher

import fitz
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# TEXT LINE STRUCTURE


class TextLine:
    def __init__(self, text, bbox=None, confidence=1.0):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence



# PREPROCESSING /TESSERACT


def preprocess(img: Image.Image) -> Image.Image:
    import cv2
    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return Image.fromarray(cv2.dilate(binary, kernel, 1))



# LOAD IMAGE / PDF


def pdf_to_image(pdf_path: Path, dpi: int = 300) -> Image.Image:
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = doc[0].get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def load_image(path: Path, dpi: int = 300) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        return pdf_to_image(path, dpi)
    return Image.open(path).convert("RGB")



# REFERENCE PARSER /SROIE FORMAT


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



# OCR: TESSERACT


def ocr_tesseract_lines(img: Image.Image, lang: str, pytesseract):
    processed = preprocess(img)

    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        config="--oem 3 --psm 3",
        output_type=pytesseract.Output.DICT
    )

    lines = []

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0

        if not text:
            continue

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        bbox = (x, y, x + w, y + h)

        lines.append(TextLine(
            text=text,
            bbox=bbox,
            confidence=conf / 100.0
        ))

    return lines



# OCR: PADDLEOCR


def ocr_paddle_lines(img: Image.Image, paddle_ocr):
    import numpy as np

    result = paddle_ocr.ocr(np.array(img), cls=True)
    lines = []

    if result and result[0]:
        for pts, (text, conf) in result[0]:
            lines.append(TextLine(
                text=text,
                bbox=pts,
                confidence=float(conf)
            ))

    return lines



# OCR ROUTER


def run_ocr(img, engine, lang, pytesseract=None, paddle_ocr=None):
    if engine == "tesseract":
        return ocr_tesseract_lines(img, lang, pytesseract)
    elif engine == "paddle":
        return ocr_paddle_lines(img, paddle_ocr)
    else:
        raise ValueError("engine must be 'tesseract' or 'paddle'")


def lines_to_text(lines):
    return " ".join([l.text for l in lines])



# METRICS


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[-1]


def cer(ref, hyp):
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return 0.0
    return min(edit_distance(ref, hyp) / len(ref), 1.0)


def wer(ref, hyp):
    ref_w = normalize(ref).split()
    hyp_w = normalize(hyp).split()
    if not ref_w:
        return 0.0
    return min(edit_distance(ref_w, hyp_w) / len(ref_w), 1.0)


def prf(ref, hyp):
    ref_w = set(normalize(ref).split())
    hyp_w = set(normalize(hyp).split())

    if not hyp_w:
        return 0.0, 0.0, 0.0

    tp = len(ref_w & hyp_w)
    p = tp / len(hyp_w)
    r = tp / len(ref_w) if ref_w else 0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0
    return p, r, f1


def similarity(ref, hyp):
    return SequenceMatcher(None, normalize(ref), normalize(hyp)).ratio()



# PAIRING DATA


IMG_EXTS = {".jpg", ".png", ".jpeg", ".tif", ".tiff", ".pdf"}


def find_pairs(images_dir: Path, refs_dir: Path):
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        ref = refs_dir / (img.stem + ".txt")
        if ref.exists():
            pairs.append((img, ref))
    return pairs



# REPORT


def fmt(x): return f"{x*100:6.2f}%"


def print_summary(results):
    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60)

    n = len(results)
    if not n:
        return

    avg = lambda k: sum(r[k] for r in results) / n

    print(f"Files: {n}")
    print(f"CER   : {fmt(avg('cer'))}")
    print(f"WER   : {fmt(avg('wer'))}")
    print(f"Prec  : {fmt(avg('p'))}")
    print(f"Recall: {fmt(avg('r'))}")
    print(f"F1    : {fmt(avg('f1'))}")
    print(f"Sim   : {fmt(avg('sim'))}")



# MAIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/task1et2_test")
    parser.add_argument("--refs", default="data/text.task1et2-test")
    parser.add_argument("--engine", choices=["tesseract", "paddle"], default="tesseract")
    parser.add_argument("--lang", default="fra")
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()

    images_dir = Path(args.images)
    refs_dir = Path(args.refs)

    pairs = find_pairs(images_dir, refs_dir)

    if not pairs:
        print("No data found")
        sys.exit(1)

    # Init engines
    pytesseract = None
    paddle_ocr = None

    if args.engine == "tesseract":
        import pytesseract
    else:
        from paddleocr import PaddleOCR
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")

    results = []

    for img_path, ref_path in pairs:
        print(f"Processing {img_path.name}")

        img = load_image(img_path, args.dpi)

        lines = run_ocr(
            img,
            args.engine,
            args.lang,
            pytesseract=pytesseract,
            paddle_ocr=paddle_ocr
        )

        hyp = lines_to_text(lines)
        ref = parse_ref_file(ref_path)

        c = cer(ref, hyp)
        w = wer(ref, hyp)
        p, r, f1 = prf(ref, hyp)
        s = similarity(ref, hyp)

        results.append({
            "cer": c,
            "wer": w,
            "p": p,
            "r": r,
            "f1": f1,
            "sim": s
        })

        print(f"CER={fmt(c)} WER={fmt(w)}")

    print_summary(results)


if __name__ == "__main__":
    main()