import argparse
import os
import sys
import time
from pathlib import Path

import fitz
from PIL import Image
import numpy as np



# PDF -> images PIL


def parse_page_range(spec: str, total: int) -> list:
    if spec.lower() == "all":
        return list(range(total))
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            indices.update(range(int(a) - 1, int(b)))
        else:
            indices.add(int(part) - 1)
    return sorted(indices)


def pdf_to_images(pdf_path: str, page_indices: list, dpi: int = 300):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    result = []
    print(f"PDF : {len(doc)} page(s) - {len(page_indices)} selectionnee(s) a {dpi} DPI")
    for idx in page_indices:
        if not (0 <= idx < len(doc)):
            print(f"  Page {idx + 1} ignoree (hors limites)")
            continue
        pix = doc[idx].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        result.append((idx + 1, img))
    doc.close()
    return result



# Pretraitement image

def preprocess(img: Image.Image) -> Image.Image:
    import cv2
    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return Image.fromarray(dilated)



# Diagnostic Tesseract

def detect_tesseract_langs(pytesseract) -> list:
    try:
        return pytesseract.get_languages()
    except Exception:
        return []


def find_tessdata(tesseract_cmd: str) -> str:
    exe = Path(tesseract_cmd)
    candidate = exe.parent / "tessdata"
    if candidate.exists():
        return str(candidate)
    return ""



# Tesseract OCR

def run_tesseract(images, lang: str = "fra", tesseract_cmd: str = None) -> list:
    try:
        import pytesseract
    except ImportError:
        print("pytesseract non installe : pip install pytesseract")
        sys.exit(1)

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        tessdata = find_tessdata(tesseract_cmd)
        if tessdata and not os.environ.get("TESSDATA_PREFIX"):
            os.environ["TESSDATA_PREFIX"] = tessdata
            print(f"  TESSDATA_PREFIX => {tessdata}")

    try:
        version = pytesseract.get_tesseract_version()
        print(f"\nTesseract {version} detecte")
    except Exception as e:
        print(f"Tesseract introuvable : {e}")
        print("  Windows : ajoutez --tesseract-cmd 'C:/Program Files/Tesseract-OCR/tesseract.exe'")
        sys.exit(1)

    available = detect_tesseract_langs(pytesseract)
    print(f"  Langues disponibles : {available}")

    chosen_lang = lang
    if lang not in available:
        fallbacks = [l for l in ["fra", "fr", "eng", "en"] if l in available]
        chosen_lang = fallbacks[0] if fallbacks else (available[0] if available else "eng")
        print(f"  Langue '{lang}' absente -> fallback : '{chosen_lang}'")
        print("  Pour installer fra : copiez fra.traineddata dans le dossier tessdata")
        print("  Telechargement : https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata")
    else:
        print(f"  Langue choisie : {chosen_lang}")

    config = "--oem 3 --psm 3"
    results = []

    for page_num, img in images:
        t0 = time.time()
        processed = preprocess(img)
        try:
            text = pytesseract.image_to_string(processed, lang=chosen_lang, config=config)
        except Exception as e:
            print(f"  Page {page_num} ERREUR : {e}")
            text = ""
        elapsed = time.time() - t0
        lines = [l for l in text.splitlines() if l.strip()]
        print(f"  Page {page_num:>3} : {len(lines):>4} ligne(s) - {elapsed:.2f}s")
        results.append({"page": page_num, "text": text.strip()})

    return results



# Formatage sortie

def format_output(results: list) -> str:
    sections = []
    for r in results:
        sep = "=" * 60
        sections.append(f"{sep}\nPage {r['page']}\n{sep}\n{r['text']}\n")
    return "\n".join(sections)



# CLI

def main():
    parser = argparse.ArgumentParser(
        description="PDF -> texte via OCR 100% local (Tesseract)"
    )
    parser.add_argument("pdf", help="Chemin vers le fichier PDF")
    parser.add_argument(
        "--lang", default="fra",
        help="Langue(s) Tesseract : fra, eng, fra+eng (defaut : fra)",
    )
    parser.add_argument(
        "--pages", default="all",
        help="Pages a traiter : '1-3', '2,4', 'all' (defaut : all)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Resolution DPI (defaut : 300)")
    parser.add_argument("--no-preprocess", action="store_true", help="Desactiver le pretraitement image")
    parser.add_argument(
        "--tesseract-cmd", default=None, metavar="PATH",
        help="Chemin vers tesseract.exe (Windows, si non dans le PATH)",
    )
    parser.add_argument("--output", default=None, help="Fichier .txt de sortie")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Fichier introuvable : {pdf_path}")
        sys.exit(1)

    tmp = fitz.open(str(pdf_path))
    total = len(tmp)
    tmp.close()

    page_indices = parse_page_range(args.pages, total)
    images = pdf_to_images(str(pdf_path), page_indices, dpi=args.dpi)

    status = "desactive" if args.no_preprocess else "active (binarisation adaptative)"
    print(f"  Pretraitement : {status}")

    results = run_tesseract(images, lang=args.lang, tesseract_cmd=args.tesseract_cmd)
    final = format_output(results)

    if args.output:
        out = Path(args.output)
        out.write_text(final, encoding="utf-8")
        print(f"\nTexte sauvegarde -> {out}")
    else:
        print("\n" + final)


if __name__ == "__main__":
    main()
