"""
Batch OCR evaluation with Tesseract + Ollama LLM judge.

Combines the metric-based evaluation (CER/WER/F1) from Comparaison_OCR.py
with the LLM extraction + judgment pipeline from ocr_llm_eval_batch.py.
Tesseract only — PaddleOCR dropped (Python 3.14 incompatibility).

USAGE
  # Basic run (metrics only, no LLM judge) :
  python scripts/ocr_eval_batch.py \\
      --docs  data/SROIE2019/0325updated.task2train(626p) \\
      --out   data/eval/results.json

  # With LLM judge (requires Ollama + mistral:7b-instruct running) :
  python scripts/ocr_eval_batch.py \\
      --docs  data/SROIE2019/0325updated.task2train(626p) \\
      --out   data/eval/results.json \\
      --llm-judge

  # Quick sample :
  python scripts/ocr_eval_batch.py \\
      --docs data/SROIE2019/0325updated.task2train(626p) \\
      --max-docs 10 --llm-judge

  # Disable contrast enhancement :
  python scripts/ocr_eval_batch.py --docs ... --no-enhance

  # Custom contrast :
  python scripts/ocr_eval_batch.py --docs ... --alpha 2.0 --beta 10
"""

import argparse
import json
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

# ── ANSI colours ─────────────────────────────────────────────────────────────

COLORS = {
    "CORRECT":   "\033[92m",
    "INCORRECT": "\033[91m",
    "PARTIEL":   "\033[93m",
    "NOT FOUND": "\033[90m",
}
RESET = "\033[0m"


def color(text: str, verdict: str) -> str:
    return f"{COLORS.get(verdict, '')}{text}{RESET}"


# ── Image loading ─────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"}


def load_image(path: Path, dpi: int = 300) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        doc  = fitz.open(str(path))
        zoom = dpi / 72
        pix  = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    return Image.open(str(path)).convert("RGB")


# ── Preprocessing ─────────────────────────────────────────────────────────────

LOW_CONTRAST_RMS = 20.0


def needs_contrast_boost(img: Image.Image,
                         threshold: float = LOW_CONTRAST_RMS) -> bool:
    """True if grayscale RMS std-dev < threshold (faded / low-contrast doc)."""
    arr = np.array(img.convert("L"), dtype=float)
    return float(arr.std()) < threshold


def enhance_contrast(img: Image.Image, alpha: float = 1.5,
                     beta: int = 0) -> Image.Image:
    """Linear contrast boost (cv2.convertScaleAbs)."""
    try:
        import cv2
    except (ImportError, AttributeError):
        return img
    img_np  = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out     = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def preprocess(img: Image.Image, enhance="auto",
               alpha: float = 1.5, beta: int = 0) -> Image.Image:
    """Contrast (adaptive) → adaptive threshold → dilate.

    enhance="auto"  apply only if needs_contrast_boost().
    enhance=True    always apply.
    enhance=False   never apply.
    """
    try:
        import cv2
    except (ImportError, AttributeError):
        return img.convert("L")
    do_enhance = needs_contrast_boost(img) if enhance == "auto" else bool(enhance)
    if do_enhance:
        img = enhance_contrast(img, alpha=alpha, beta=beta)
    arr    = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return Image.fromarray(cv2.dilate(binary, kernel, iterations=1))


# ── OCR ───────────────────────────────────────────────────────────────────────

def run_ocr(img: Image.Image, lang: str, enhance="auto",
            alpha: float = 1.5, beta: int = 0) -> str:
    """Returns OCR text as a single string."""
    processed = preprocess(img, enhance=enhance, alpha=alpha, beta=beta)
    data = pytesseract.image_to_data(
        processed, lang=lang,
        config="--oem 3 --psm 3",
        output_type=pytesseract.Output.DICT,
    )
    words = [data["text"][i].strip() for i in range(len(data["text"]))
             if data["text"][i].strip()]
    return "\n".join(words)


# ── Metrics ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def edit_distance(a, b) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, len(b) + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] \
                    else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[-1]


def cer(ref, hyp) -> float:
    r, h = normalize(ref), normalize(hyp)
    return 0.0 if not r else min(edit_distance(r, h) / len(r), 1.0)


def wer(ref, hyp) -> float:
    rw, hw = normalize(ref).split(), normalize(hyp).split()
    return 0.0 if not rw else min(edit_distance(rw, hw) / len(rw), 1.0)


def prf(ref, hyp):
    rw, hw = set(normalize(ref).split()), set(normalize(hyp).split())
    if not hw:
        return 0.0, 0.0, 0.0
    tp = len(rw & hw)
    p  = tp / len(hw)
    r  = tp / len(rw) if rw else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def similarity(ref, hyp) -> float:
    return SequenceMatcher(None, normalize(ref), normalize(hyp)).ratio()


# ── Reference parsers ─────────────────────────────────────────────────────────

def parse_sroie_txt(ref_path: Path) -> str:
    """SROIE task-1 format : 8 bbox coords + text."""
    lines = []
    with open(ref_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            text  = ",".join(parts[8:]) if len(parts) >= 9 else line
            if text:
                lines.append(text)
    return "\n".join(lines)


def parse_json_ref(ref_path: Path) -> dict:
    """SROIE task-2 JSON format : {company, date, address, total}."""
    with open(ref_path, encoding="utf-8") as f:
        return json.load(f)


# ── Ollama helpers ────────────────────────────────────────────────────────────

# Extraction prompt — structured output, Ollama format=json
EXTRACT_PROMPT = """\
Tu es un assistant d'analyse de tickets de caisse et factures.
Voici le texte extrait par OCR d'un document :

{ocr_text}

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou après,
avec exactement ces trois clés :
{{
  "total":     "<montant numérique tel qu'il apparaît sur le document, ex: 33.90, ou NOT FOUND>",
  "company":   "<nom de l'entreprise tel qu'il apparaît sur le document, ou NOT FOUND>",
  "date":      "<date EXACTEMENT telle qu'elle apparaît sur le document, ex: 25/12/2018 ou 18-11-18, ou NOT FOUND>"
}}
Règles strictes :
- Ne jamais inventer ou reformater une valeur absente du texte OCR.
- Pour "total" : prendre le montant final à payer (TOTAL, GRAND TOTAL, AMOUNT DUE).
  Si le plus grand chiffre visible semble être un paiement en espèces, le prendre.
- Pour "date" : recopier la date exactement comme dans le texte, sans la convertir.
- Si une information n'est pas présente dans le texte OCR, écrire NOT FOUND.\
"""

# Judge prompt — structured output, Ollama format=json
JUDGE_PROMPT = """\
Tu es un vérificateur de qualité pour un système d'extraction d'informations.
Valeurs de référence (ground truth) :
  total     : {ref_total}
  company   : {ref_company}
  date      : {ref_date}

Valeurs extraites :
  total     : {ext_total}
  company   : {ext_company}
  date      : {ext_date}

Pour chaque champ, indique CORRECT, PARTIEL ou INCORRECT selon ces règles strictes :

Règle "total" : convertis les deux valeurs en nombre flottant (ex: 80.9 == 80.90).
  CORRECT si les valeurs numériques sont égales. INCORRECT sinon.
  Si la valeur extraite est NOT FOUND : INCORRECT.

Règle "company" : CORRECT si le nom principal de l'entreprise est présent
  (la forme juridique comme SDN BHD, la ville ou l'adresse peuvent être absentes).
  PARTIEL si au moins 60 % des mots-clés du nom de référence sont présents.
  INCORRECT si le nom principal est absent ou erroné.

Règle "date" : compare uniquement les valeurs numériques de jour, mois et année,
  quel que soit le format (25/12/2018 == 25-12-18 == 2018-12-25).
  CORRECT si jour + mois + année correspondent tous les trois.
  INCORRECT si l'une des trois valeurs diffère, même légèrement.
  Si la valeur extraite est NOT FOUND : INCORRECT.

Réponds UNIQUEMENT avec un objet JSON valide :
{{
  "total":   {{"verdict": "<CORRECT|PARTIEL|INCORRECT>", "reason": "<court>"}},
  "company": {{"verdict": "<CORRECT|PARTIEL|INCORRECT>", "reason": "<court>"}},
  "date":    {{"verdict": "<CORRECT|PARTIEL|INCORRECT>", "reason": "<court>"}}
}}\
"""


def _ollama_generate(host: str, model: str, prompt: str,
                     max_tokens: int = 300) -> str:
    try:
        import ollama
    except ImportError as exc:
        raise ImportError("pip install ollama") from exc
    client = ollama.Client(host=host)
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0, "num_predict": max_tokens},
    )
    return resp["message"]["content"].strip()


def extract_info(host: str, model: str, ocr_text: str) -> dict:
    prompt = EXTRACT_PROMPT.format(ocr_text=ocr_text[:3000])
    raw    = _ollama_generate(host, model, prompt)
    try:
        data = json.loads(raw)
        return {
            "total":   str(data.get("total",   "NOT FOUND")).strip() or "NOT FOUND",
            "company": str(data.get("company", "NOT FOUND")).strip() or "NOT FOUND",
            "date":    str(data.get("date",    "NOT FOUND")).strip() or "NOT FOUND",
        }
    except json.JSONDecodeError:
        return {"total": "NOT FOUND", "company": "NOT FOUND", "date": "NOT FOUND"}


def judge_info(host: str, model: str,
               extracted: dict, reference: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        ref_total=reference.get("total",   "?"),
        ref_company=reference.get("company", "?"),
        ref_date=reference.get("date",    "?"),
        ext_total=extracted["total"],
        ext_company=extracted["company"],
        ext_date=extracted["date"],
    )
    raw = _ollama_generate(host, model, prompt, max_tokens=400)
    try:
        data = json.loads(raw)
        return {
            field: data[field]["verdict"].upper()
            for field in ("total", "company", "date")
            if field in data and "verdict" in data[field]
        }
    except (json.JSONDecodeError, KeyError, AttributeError):
        return {}


# ── Auto comparison (no LLM needed) ──────────────────────────────────────────

def _norm_amount(s: str) -> str:
    nums = re.findall(r"\d+\.?\d*", s.replace(",", "."))
    if not nums:
        return ""
    try:
        # Normalise via float so 80.9 == 80.90, 9.00 == 9.0, etc.
        return str(float(nums[-1]))
    except ValueError:
        return nums[-1]


def _norm_date(s: str) -> str:
    return re.sub(r"[\s\-/\.]", "", s).lower()


def _norm_company(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def compare_field(field: str, extracted: str, reference: str) -> str:
    if extracted.upper() == "NOT FOUND":
        return "NOT FOUND"
    if field == "total":
        return "CORRECT" if _norm_amount(extracted) == _norm_amount(reference) \
               else "INCORRECT"
    if field == "date":
        return "CORRECT" if _norm_date(extracted) == _norm_date(reference) \
               else "INCORRECT"
    if field == "company":
        e, r = _norm_company(extracted), _norm_company(reference)
        if e == r:
            return "CORRECT"
        rw, ew = set(r.split()), set(e.split())
        if rw and len(rw & ew) / len(rw) >= 0.6:
            return "PARTIEL"
        return "INCORRECT"
    return "INCORRECT"


# ── File pairing ──────────────────────────────────────────────────────────────

def find_pairs(docs_dir: Path) -> list:
    """
    Pairs images with their JSON ground-truth (task-2 format).
    Falls back to SROIE .txt if no JSON exists.
    """
    pairs = []
    for img_path in sorted(docs_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        json_ref = docs_dir / (img_path.stem + ".txt")
        if not json_ref.exists():
            continue
        try:
            ref = json.loads(json_ref.read_text(encoding="utf-8"))
            pairs.append((img_path, ref, "json"))
        except json.JSONDecodeError:
            # Plain SROIE task-1 text file — no structured ref for LLM judge
            pairs.append((img_path, None, "txt"))
    return pairs


# ── Display ───────────────────────────────────────────────────────────────────

def _fmt(x: float) -> str:
    return f"{x * 100:6.2f}%"


def print_doc_result(name, extracted, reference, auto, llm_verdicts, final):
    sep = "─" * 75
    print(f"\n{sep}")
    print(f"  {name}")
    print(f"  {'FIELD':<12} {'REFERENCE':<22} {'EXTRACTED':<22} {'AUTO':<12} "
          f"{'LLM':<12} FINAL")
    print(f"  {'-'*12} {'-'*22} {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
    for field in ("total", "company", "date"):
        ref_v  = str(reference.get(field, "?"))[:21]
        ext_v  = str(extracted.get(field, "?"))[:21]
        auto_v = auto.get(field, "?")
        llm_v  = llm_verdicts.get(field, "—")
        fin_v  = final.get(field, "?")
        print(f"  {field.upper():<12} {ref_v:<22} {ext_v:<22} "
              f"{color(auto_v, auto_v):<12} {color(llm_v, llm_v):<12} "
              f"{color(fin_v, fin_v)}")


def print_stats(results: list) -> None:
    n = len(results)
    if not n:
        return
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  GLOBAL STATS  ({n} documents)")
    print(sep)
    for field in ("total", "company", "date"):
        counts = {"CORRECT": 0, "PARTIEL": 0, "INCORRECT": 0, "NOT FOUND": 0}
        for r in results:
            v = r.get("comparisons", {}).get(field, "INCORRECT")
            counts[v] = counts.get(v, 0) + 1
        cr = counts["CORRECT"]   / n * 100
        pr = counts["PARTIEL"]   / n * 100
        ir = counts["INCORRECT"] / n * 100
        nr = counts["NOT FOUND"] / n * 100
        print(f"\n  {field.upper()}")
        print(f"    Correct   : " + color(f"{counts['CORRECT']:3d} ({cr:5.1f}%)", "CORRECT"))
        print(f"    Partiel   : " + color(f"{counts['PARTIEL']:3d} ({pr:5.1f}%)", "PARTIEL"))
        print(f"    Incorrect : " + color(f"{counts['INCORRECT']:3d} ({ir:5.1f}%)", "INCORRECT"))
        print(f"    Not found : {counts['NOT FOUND']:3d} ({nr:5.1f}%)")

    score = sum(
        (r.get("comparisons", {}).get("total",   "") == "CORRECT") +
        (r.get("comparisons", {}).get("company", "") == "CORRECT") +
        (r.get("comparisons", {}).get("date",    "") == "CORRECT") +
        0.5 * (r.get("comparisons", {}).get("total",   "") == "PARTIEL") +
        0.5 * (r.get("comparisons", {}).get("company", "") == "PARTIEL") +
        0.5 * (r.get("comparisons", {}).get("date",    "") == "PARTIEL")
        for r in results
    )
    print(f"\n  Global score : {score:.1f} / {n * 3}  ({score / (n * 3) * 100:.1f}%)")
    print(sep)

    # Metric averages (when available)
    metric_results = [r for r in results if "metrics" in r]
    if metric_results:
        m = len(metric_results)
        avg = lambda k: sum(r["metrics"][k] for r in metric_results) / m
        print(f"\n  OCR METRICS  ({m} docs with SROIE refs)")
        print(f"    CER    : {_fmt(avg('cer'))}")
        print(f"    WER    : {_fmt(avg('wer'))}")
        print(f"    F1     : {_fmt(avg('f1'))}")
        print(f"    Sim    : {_fmt(avg('sim'))}")
        print(sep)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch Tesseract OCR + Ollama LLM evaluation"
    )
    parser.add_argument("--docs",       required=True,
                        help="Directory with images + JSON ground-truth")
    parser.add_argument("--out",        default=None,
                        help="Output JSON results file")
    parser.add_argument("--lang",       default="fra+eng")
    parser.add_argument("--dpi",        type=int,   default=300)
    parser.add_argument("--max-docs",   type=int,   default=None)
    parser.add_argument("--enhance",    default="auto",
                        choices=["auto", "on", "off"],
                        help="Contrast filter: auto (default) = apply only on "
                             "low-contrast images, on = always, off = never")
    parser.add_argument("--alpha",      type=float, default=1.5,
                        help="Contrast multiplier (default 1.5)")
    parser.add_argument("--beta",       type=int,   default=0,
                        help="Brightness offset (default 0)")
    parser.add_argument("--llm-judge",  action="store_true",
                        help="Enable LLM extraction + judgment via Ollama")
    parser.add_argument("--ollama-host",  default=os.environ.get(
                        "DOQMENT_OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--ollama-model", default=os.environ.get(
                        "DOQMENT_OLLAMA_TEXT_MODEL", "mistral:7b-instruct"))
    args = parser.parse_args()

    enhance_mode = {"auto": "auto", "on": True, "off": False}[args.enhance]

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        sys.exit(f"[ERROR] Directory not found: {docs_dir}")

    pairs = find_pairs(docs_dir)
    if not pairs:
        sys.exit("[ERROR] No image+JSON pairs found.")

    if args.max_docs:
        pairs = pairs[:args.max_docs]

    print(f"\n  {len(pairs)} document(s)  |  "
          f"enhance={args.enhance} (α={args.alpha} β={args.beta})  |  "
          f"LLM judge={'on' if args.llm_judge else 'off'}")

    results = []

    for i, (img_path, reference, ref_type) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {img_path.name}")

        # 1. OCR
        try:
            img      = load_image(img_path, args.dpi)
            ocr_text = run_ocr(img, args.lang,
                               enhance=enhance_mode,
                               alpha=args.alpha, beta=args.beta)
        except Exception as exc:
            print(f"  [WARN] OCR failed: {exc}")
            continue

        record = {"file": img_path.name, "reference": reference or {}}

        # 2. LLM extraction + judgment (only for JSON refs)
        auto_comparisons = {}
        llm_verdicts     = {}
        comparisons      = {}

        if args.llm_judge and reference is not None:
            extracted         = extract_info(args.ollama_host, args.ollama_model,
                                             ocr_text)
            auto_comparisons  = {
                f: compare_field(f, extracted[f], reference.get(f, ""))
                for f in ("total", "company", "date")
            }
            llm_verdicts = judge_info(args.ollama_host, args.ollama_model,
                                      extracted, reference)
            comparisons  = {
                f: llm_verdicts.get(f) or auto_comparisons[f]
                for f in ("total", "company", "date")
            }
            print_doc_result(img_path.name, extracted, reference,
                             auto_comparisons, llm_verdicts, comparisons)
            record.update({
                "extracted":        extracted,
                "auto_comparisons": auto_comparisons,
                "llm_verdicts":     llm_verdicts,
                "comparisons":      comparisons,
            })
        else:
            # No LLM : only store raw OCR text
            print(f"  OCR chars: {len(ocr_text)}")
            record["ocr_text"] = ocr_text[:500]

        results.append(record)

    if args.llm_judge:
        print_stats(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
