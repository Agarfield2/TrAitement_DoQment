"""
USAGE

  python ocr_llm_eval_batch.py \
      --docs   "data/0325updated.task2train(626p)" \
      --model  models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
      --out    eval/results_batch.json

  # Petit échantillon pour tester :
  python ocr_llm_eval_batch.py \
      --docs   "data/0325updated.task2train(626p)" \
      --model  models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
      --max-docs 10 \
      --out    eval/results_batch.json
"""

import argparse
import json
import sys
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import fitz


# Constantes 

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"}

COLORS = {
    "CORRECT":      "\033[92m",
    "INCORRECT":    "\033[91m",
    "PARTIEL":      "\033[93m",
    "NOT FOUND":    "\033[90m",
}
RESET = "\033[0m"

def color(text: str, verdict: str) -> str:
    return f"{COLORS.get(verdict, '')}{text}{RESET}"


# Filtre contraste 

def enhance_contrast(img: Image.Image, alpha: float = 1.5, beta: int = 0) -> Image.Image:
    """
    Rehausse le contraste via convertScaleAbs.
    alpha : contraste (1.0 = inchangé, >1 = plus contrasté)
    beta  : luminosité (+/- offset)
    """
    img_np = np.array(img)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    enhanced = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced)


# Chargement image / PDF 

def load_image(path: Path, dpi: int = 300) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        zoom = dpi / 72
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    return Image.open(str(path)).convert("RGB")


# OCR PaddleOCR 

def init_ocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        sys.exit("[ERREUR] pip install paddlepaddle paddleocr")
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    print("  [OCR] Chargement PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    print("  [OCR] Prêt")
    return ocr


def run_ocr(ocr, img: Image.Image, enhance: bool = True,
            alpha: float = 1.5, beta: int = 0) -> str:
    if enhance:
        img = enhance_contrast(img, alpha=alpha, beta=beta)
    result = ocr.ocr(np.array(img), cls=True)
    lines = []
    if result and result[0]:
        for pts, (text, conf) in result[0]:
            lines.append(text)
    return "\n".join(lines)


# Chargement Mistral

def init_mistral(model_path: str, n_gpu_layers: int = 0):
    try:
        from llama_cpp import Llama
    except ImportError:
        sys.exit("[ERREUR] pip install llama-cpp-python")
    if not Path(model_path).exists():
        sys.exit(f"[ERREUR] Modèle introuvable : {model_path}")
    print(f"  [LLM] Chargement Mistral...")
    llm = Llama(model_path=model_path, n_ctx=4096,
                n_gpu_layers=n_gpu_layers, verbose=False)
    print("  [LLM] Prêt")
    return llm


def ask_mistral(llm, prompt: str, max_tokens: int = 256) -> str:
    response = llm(
        prompt, max_tokens=max_tokens, temperature=0.1,
        echo=False, stop=["</s>", "[INST]"],
    )
    return response["choices"][0]["text"].strip()


#  Étape 1 : extraction ──

EXTRACT_PROMPT = """<s>[INST] Tu es un assistant d'analyse de tickets de caisse et factures.
Voici le texte extrait par OCR d'un document :

{ocr_text}

Réponds UNIQUEMENT dans ce format exact, une information par ligne :
TOTAL: <montant>
ENTREPRISE: <nom>
DATE: <date>
(fait attention, le plus grand chifre peut aussi etre se qui a été payer en cash !)
Si une information est absente écris NOT FOUND. [/INST]"""


def extract_info(llm, ocr_text: str) -> dict:
    prompt = EXTRACT_PROMPT.format(ocr_text=ocr_text[:3000])
    raw = ask_mistral(llm, prompt)
    result = {"total": "NOT FOUND", "company": "NOT FOUND", "date": "NOT FOUND"}
    mapping = {"TOTAL": "total", "ENTREPRISE": "company", "DATE": "date"}
    for line in raw.splitlines():
        for prefix, key in mapping.items():
            if line.strip().upper().startswith(prefix + ":"):
                val = line.split(":", 1)[1].strip()
                if val:
                    result[key] = val
    return result


#  Étape 2 : comparaison avec ground truth 

def normalize_amount(s: str) -> str:
    """Extrait uniquement les chiffres et le point décimal."""
    s = s.replace(",", ".")
    nums = re.findall(r"\d+\.?\d*", s)
    return nums[-1] if nums else ""


def normalize_date(s: str) -> str:
    """Normalise les dates en retirant séparateurs et espaces."""
    return re.sub(r"[\s\-/\.]", "", s).lower()


def normalize_company(s: str) -> str:
    """Minuscules, retire ponctuation et espaces superflus."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compare_field(field: str, extracted: str, reference: str) -> str:
    """Retourne CORRECT / PARTIEL / INCORRECT / NOT FOUND."""
    if extracted.upper() == "NOT FOUND":
        return "NOT FOUND"

    if field == "total":
        e = normalize_amount(extracted)
        r = normalize_amount(reference)
        return "CORRECT" if e == r else "INCORRECT"

    elif field == "date":
        e = normalize_date(extracted)
        r = normalize_date(reference)
        return "CORRECT" if e == r else "INCORRECT"

    elif field == "company":
        e = normalize_company(extracted)
        r = normalize_company(reference)
        if e == r:
            return "CORRECT"
        # Partiel si au moins 60% des mots de la référence sont présents
        ref_words = set(r.split())
        ext_words = set(e.split())
        if ref_words and len(ref_words & ext_words) / len(ref_words) >= 0.6:
            return "PARTIEL"
        return "INCORRECT"

    return "INCORRECT"


# Étape 3 : Mistral juge

JUDGE_PROMPT = """<s>[INST] Tu es un vérificateur de qualité pour un système d'extraction d'informations.
Voici les valeurs de référence (ground truth) d'un reçu :
- Total attendu    : {ref_total}
- Entreprise attendue : {ref_company}
- Date attendue    : {ref_date}

Voici ce qu'a extrait le système :
- Total extrait    : {ext_total}
- Entreprise extraite : {ext_company}
- Date extraite    : {ext_date}

Pour chaque champ, réponds CORRECT, PARTIEL ou INCORRECT avec une raison courte.
exemple :
MR D.I.Y. (JOHOR) SDN BH pour MR DIY est correst car contient le mot clé principal, car pas forcement besoin d'avoir la partie (JOHOR) qui est le lieu, ou SDN BH qui est la forme juridique. Par contre si c'était juste "JOHOR" ça serait incorrect car c'est pas l'entreprise.
Format exact :
TOTAL: <CORRECT|PARTIEL|INCORRECT> - <raison>
ENTREPRISE: <CORRECT|PARTIEL|INCORRECT> - <raison>
DATE: <CORRECT|PARTIEL|INCORRECT> - <raison> [/INST]"""


def judge_info(llm, extracted: dict, reference: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        ref_total=reference.get("total", "?"),
        ref_company=reference.get("company", "?"),
        ref_date=reference.get("date", "?"),
        ext_total=extracted["total"],
        ext_company=extracted["company"],
        ext_date=extracted["date"],
    )
    raw = ask_mistral(llm, prompt, max_tokens=300)
    verdicts = {}
    for line in raw.splitlines():
        for key, label in [("TOTAL", "total"), ("ENTREPRISE", "company"), ("DATE", "date")]:
            if line.strip().upper().startswith(key + ":"):
                verdicts[label] = line.split(":", 1)[1].strip()
    return verdicts


# Appariement fichiers

def find_pairs(docs_dir: Path) -> list[tuple[Path, dict]]:
    pairs = []
    for img_path in sorted(docs_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        ref_path = docs_dir / (img_path.stem + ".txt")
        if not ref_path.exists():
            continue
        try:
            with open(ref_path, encoding="utf-8") as f:
                ref = json.load(f)
            pairs.append((img_path, ref))
        except json.JSONDecodeError:
            print(f"  [WARN] JSON invalide : {ref_path.name}")
    return pairs


# Affichage par document

def print_doc_result(img_name: str, extracted: dict, reference: dict,
                     auto_comparisons: dict, llm_verdicts: dict, comparisons: dict):
    sep = "─" * 75
    print(f"\n{sep}")
    print(f"  {img_name}")
    print(f"  {'CHAMP':<12} {'RÉFÉRENCE':<22} {'EXTRAIT':<22} {'AUTO':<12} {'LLM':<12} {'FINAL'}")
    print(f"  {'-'*12} {'-'*22} {'-'*22} {'-'*12} {'-'*12} {'-'*10}")
    for field in ("total", "company", "date"):
        ref_val   = reference.get(field, "?")[:21]
        ext_val   = extracted.get(field, "?")[:21]
        auto_v    = auto_comparisons.get(field, "?")
        llm_raw   = llm_verdicts.get(field, "—")
        # version courte du verdict LLM (juste le mot)
        llm_short = llm_raw.split("-")[0].strip()[:10] if llm_raw != "—" else "—"
        final_v   = comparisons.get(field, "?")
        source    = "(LLM)" if field in llm_verdicts and llm_verdicts[field] else "(AUTO)"
        print(f"  {field.upper():<12} {ref_val:<22} {ext_val:<22} "
              f"{color(auto_v, auto_v):<12} {color(llm_short, llm_short):<12} "
              f"{color(final_v, final_v)} {source}")


# Stats globales

def print_stats(results: list[dict]):
    n = len(results)
    if not n:
        return

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  STATS GLOBALES  ({n} documents)")
    print(sep)

    for field in ("total", "company", "date"):
        counts = {"CORRECT": 0, "PARTIEL": 0, "INCORRECT": 0, "NOT FOUND": 0}
        for r in results:
            v = r["comparisons"].get(field, "INCORRECT")
            counts[v] = counts.get(v, 0) + 1

        correct_rate   = counts["CORRECT"]   / n * 100
        partiel_rate   = counts["PARTIEL"]   / n * 100
        incorrect_rate = counts["INCORRECT"] / n * 100
        notfound_rate  = counts["NOT FOUND"] / n * 100

        correct_str   = f"{counts['CORRECT']:3d} ({correct_rate:5.1f}%)"
        partiel_str   = f"{counts['PARTIEL']:3d} ({partiel_rate:5.1f}%)"
        incorrect_str = f"{counts['INCORRECT']:3d} ({incorrect_rate:5.1f}%)"

        print(f"\n  {field.upper()}")
        print(f"    Correct    : {color(correct_str, 'CORRECT')}")
        print(f"    Partiel    : {color(partiel_str, 'PARTIEL')}")
        print(f"    Incorrect  : {color(incorrect_str, 'INCORRECT')}")
        print(f"    Not found  : {counts['NOT FOUND']:3d} ({notfound_rate:5.1f}%)")

    # Score global = (CORRECT + 0.5*PARTIEL) sur les 3 champs
    total_score = sum(
        (r["comparisons"].get("total",   "INCORRECT") == "CORRECT") +
        (r["comparisons"].get("company", "INCORRECT") == "CORRECT") +
        (r["comparisons"].get("date",    "INCORRECT") == "CORRECT") +
        0.5 * (r["comparisons"].get("total",   "") == "PARTIEL") +
        0.5 * (r["comparisons"].get("company", "") == "PARTIEL") +
        0.5 * (r["comparisons"].get("date",    "") == "PARTIEL")
        for r in results
    )
    print(f"\n  Score global   : {total_score:.1f} / {n*3}  "
          f"({total_score/(n*3)*100:.1f}%)")
    print(sep)


# Main 

def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR + Mistral extraction + comparaison JSON ground truth"
    )
    parser.add_argument("--docs",       required=True,  help="Dossier images + .txt JSON")
    parser.add_argument("--model",      required=True,  help="Chemin .gguf Mistral")
    parser.add_argument("--out",        default=None,   help="Fichier JSON résultats")
    parser.add_argument("--max-docs",   type=int, default=None)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--dpi",        type=int, default=300)
    parser.add_argument("--no-enhance",   action="store_true",
                        help="Désactive le filtre de contraste")
    parser.add_argument("--alpha",        type=float, default=1.5,
                        help="Intensité du contraste (défaut 1.5)")
    parser.add_argument("--beta",         type=int,   default=0,
                        help="Offset de luminosité (défaut 0)")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Désactive le jugement LLM (plus rapide, stats auto uniquement)")
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        sys.exit(f"[ERREUR] Dossier introuvable : {docs_dir}")

    # Appariement image ↔ JSON
    pairs = find_pairs(docs_dir)
    if not pairs:
        sys.exit("[ERREUR] Aucune paire image+JSON trouvée.")

    if args.max_docs:
        pairs = pairs[:args.max_docs]

    print(f"\n  {len(pairs)} document(s) à traiter")

    # Init modèles
    ocr = init_ocr()
    llm = init_mistral(args.model, args.gpu_layers)

    results = []

    for i, (img_path, reference) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {img_path.name}")

        # 1. OCR
        try:
            img      = load_image(img_path, args.dpi)
            ocr_text = run_ocr(ocr, img,
                               enhance=not args.no_enhance,
                               alpha=args.alpha,
                               beta=args.beta)
        except Exception as e:
            print(f"  [WARN] OCR échoué : {e}")
            continue

        # 2. Extraction Mistral
        extracted = extract_info(llm, ocr_text)

        # 3. Comparaison automatique
        auto_comparisons = {
            field: compare_field(field, extracted[field], reference.get(field, ""))
            for field in ("total", "company", "date")
        }

        # 4. Jugement LLM
        llm_verdicts = {}
        if not args.no_llm_judge:
            llm_verdicts = judge_info(llm, extracted, reference)

        # 5. Verdict final
        def parse_llm_verdict(raw: str) -> str:
            raw_up = raw.upper()
            if "INCORRECT" in raw_up:
                return "INCORRECT"
            if "PARTIEL" in raw_up:
                return "PARTIEL"
            if "CORRECT" in raw_up:
                return "CORRECT"
            return "INCORRECT"

        comparisons = {}
        for field in ("total", "company", "date"):
            if field in llm_verdicts and llm_verdicts[field]:
                comparisons[field] = parse_llm_verdict(llm_verdicts[field])
            else:
                comparisons[field] = auto_comparisons[field]

        # 5. Affichage
        print_doc_result(img_path.name, extracted, reference,
                         auto_comparisons, llm_verdicts, comparisons)

        results.append({
            "file":            img_path.name,
            "reference":       reference,
            "extracted":       extracted,
            "auto_comparisons": auto_comparisons,
            "llm_verdicts":    llm_verdicts,
            "comparisons":     comparisons,
        })

    # Stats finales
    print_stats(results)

    # Sauvegarde
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Résultats sauvegardés → {out_path}")


if __name__ == "__main__":
    main()