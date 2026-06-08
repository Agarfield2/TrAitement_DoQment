"""
Évaluation par lot sur SROIE-Dataset_v2 — Phase 1 (Tesseract + LLM texte)
                                          et Phase 2 (Qwen2.5-VL vision).

Structure attendue du dataset :
  <dataset>/
  ├── test/
  │   ├── img/        ← images .jpg / .png des reçus
  │   ├── box/        ← annotations ICDAR (x1,y1,...,x4,y4,texte)
  │   └── entities/   ← JSON ground-truth {company, date, address, total}
  └── train/          ← même structure (optionnel)

Pour chaque reçu on pose trois questions au modèle :
  1. Nom de la companie (company)
  2. Montant total (total)
  3. Date (date)

Puis on compare la réponse extraite à la ground-truth et on compte
le nombre de reçus dont on a réussi à extraire correctement les 3 champs.

USAGE — Phase 1 (Tesseract OCR → Mistral texte) :
  python scripts/ocr_eval_batch.py \\
      --pipeline phase1 \\
      --dataset  data/SROIE-Dataset_v2 \\
      --split    test \\
      --out      data/eval/results_phase1.json

USAGE — Phase 2 (image → Qwen2.5-VL vision, pas de Tesseract) :
  python scripts/ocr_eval_batch.py \\
      --pipeline phase2 \\
      --dataset  data/SROIE-Dataset_v2 \\
      --split    test \\
      --out      data/eval/results_phase2.json

Options communes :
  --max-docs N          Limiter à N reçus
  --enhance auto|on|off Filtre contraste Tesseract (Phase 1 uniquement)
  --use-box-annotations Utiliser box/ au lieu de Tesseract (Phase 1 uniquement)
  --ollama-host URL     Hôte Ollama (défaut: DOQMENT_OLLAMA_HOST ou localhost:11434)
  --ollama-model TAG    Modèle texte Phase 1 (défaut: DOQMENT_OLLAMA_TEXT_MODEL)
  --vision-model TAG    Modèle vision Phase 2 (défaut: DOQMENT_OLLAMA_VISION_MODEL)

DocVQA (questions libres + métrique ANLS) — Task-1 single-page et Task-3 infographics :
  python scripts/ocr_eval_batch.py \\
      --dataset-type docvqa \\
      --dataset      data/DocVQA \\
      --task         both \\
      --split        val \\
      --pipeline     phase1 \\
      --out          data/eval/results_docvqa_phase1.json

  Structure attendue (par tâche) :
    data/DocVQA/Task-1_.../  {Annotations/<split>_v1.0_withQT.json, Images/, OCR/}
    data/DocVQA/Task-3_.../  {Annotations/infographicsVQA_<split>_v1.0_withQT.json, Images/, OCR/}

  Options DocVQA :
    --task task1|task3|both   Tâche(s) à évaluer (défaut: both)
    --use-provided-ocr        Phase 1 : utiliser OCR/ fourni au lieu de Tesseract
    --anls-threshold X        Seuil ANLS (défaut: 0.5)
  Note : le split 'test' n'a pas de réponses publiques → utiliser 'val' (défaut) ou 'train'.
"""

import argparse
import base64
import io
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

# Permet d'importer le paquet doqment (OCR docTR) quand le script est lance
# directement (python scripts/ocr_eval_batch.py) depuis la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "DOQMENT_TESSERACT_PATH", "/usr/bin/tesseract"
)

# ── ANSI couleurs ─────────────────────────────────────────────────────────────

COLORS = {
    "CORRECT":   "\033[92m",
    "INCORRECT": "\033[91m",
    "PARTIEL":   "\033[93m",
    "NOT FOUND": "\033[90m",
}
RESET = "\033[0m"


def color(text: str, verdict: str) -> str:
    return f"{COLORS.get(verdict, '')}{text}{RESET}"


# ── Chargement image ──────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"}


def load_image(path: Path, dpi: int = 300) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        zoom = dpi / 72
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    return Image.open(str(path)).convert("RGB")


# ── Pré-traitement (Phase 1) ──────────────────────────────────────────────────

LOW_CONTRAST_RMS = 20.0


def needs_contrast_boost(img: Image.Image,
                         threshold: float = LOW_CONTRAST_RMS) -> bool:
    arr = np.array(img.convert("L"), dtype=float)
    return float(arr.std()) < threshold


def enhance_contrast(img: Image.Image, alpha: float = 1.5,
                     beta: int = 0) -> Image.Image:
    try:
        import cv2
    except (ImportError, AttributeError):
        return img
    img_np = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def preprocess(img: Image.Image, enhance="auto",
               alpha: float = 1.5, beta: int = 0) -> Image.Image:
    try:
        import cv2
    except (ImportError, AttributeError):
        return img.convert("L")
    do_enhance = needs_contrast_boost(img) if enhance == "auto" else bool(enhance)
    if do_enhance:
        img = enhance_contrast(img, alpha=alpha, beta=beta)
    arr = np.array(img.convert("L"))
    binary = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return Image.fromarray(cv2.dilate(binary, kernel, iterations=1))


# ── OCR Tesseract (Phase 1) ───────────────────────────────────────────────────

def run_tesseract(img: Image.Image, lang: str, enhance="auto",
                  alpha: float = 1.5, beta: int = 0) -> str:
    """Retourne le texte OCR Tesseract sous forme de chaîne."""
    processed = preprocess(img, enhance=enhance, alpha=alpha, beta=beta)
    try:
        available = pytesseract.get_languages()
        parts = [p for p in lang.split("+") if p in available]
        chosen = "+".join(parts) if parts else (available[0] if available else "eng")
    except Exception:
        chosen = lang
    data = pytesseract.image_to_data(
        processed, lang=chosen,
        config="--oem 3 --psm 3",
        output_type=pytesseract.Output.DICT,
    )
    words = [data["text"][i].strip() for i in range(len(data["text"]))
             if data["text"][i].strip()]
    return "\n".join(words)


def read_box_annotation(box_path: Path) -> str:
    """Lit un fichier box/ ICDAR et retourne le texte ligne par ligne."""
    lines = []
    with open(box_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(",", 8)
            text = parts[8].strip() if len(parts) >= 9 else raw
            if text:
                lines.append(text)
    return "\n".join(lines)


# ── Selecteur de moteur OCR (docTR par defaut, comme le reste du projet) ──────

def _project_default_ocr_engine() -> str:
    """Moteur OCR par defaut du projet (doqment.settings.ocr_engine, sinon docTR)."""
    try:
        from doqment.settings import load_settings
        return load_settings().ocr_engine
    except Exception:
        return "doctr"


def run_ocr_text(img: Image.Image, args) -> str:
    """
    Retourne le texte OCR selon args.ocr_engine.

    - "doctr" (defaut) : delegue a doqment.ocr.ocr_image (modele docTR) et
      concatene les lignes detectees.
    - "tesseract"      : voie Tesseract locale (avec --lang/--enhance/--alpha/--beta).
    """
    if args.ocr_engine == "tesseract":
        enhance_mode = {"auto": "auto", "on": True, "off": False}[args.enhance]
        return run_tesseract(img, args.lang, enhance=enhance_mode,
                             alpha=args.alpha, beta=args.beta)
    # docTR
    from doqment.ocr import ocr_image
    lines = ocr_image(img, engine="doctr")
    return "\n".join(ln.text for ln in lines)


def ocr_engine_label(args) -> str:
    """Libelle lisible du moteur OCR pour le resume de configuration."""
    if args.ocr_engine == "tesseract":
        return f"Tesseract ({args.lang})"
    return "docTR"


# ── Métriques textuelles ──────────────────────────────────────────────────────

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


# ── Prompts Ollama ────────────────────────────────────────────────────────────

# Phase 1 : extraction depuis texte OCR
EXTRACT_PROMPT_TEXT = """\
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
- Pour "date" : recopier la date exactement comme dans le texte, sans la convertir.
- Si une information n'est pas présente dans le texte OCR, écrire NOT FOUND.\
"""

# Phase 2 : extraction directement depuis l'image (pas de texte OCR)
EXTRACT_PROMPT_VISION = """\
Tu es un assistant d'analyse de tickets de caisse et factures.
Regarde attentivement l'image du reçu fournie.

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou après,
avec exactement ces trois clés :
{
  "total":     "<montant numérique final tel qu'il apparaît sur le reçu, ex: 33.90, ou NOT FOUND>",
  "company":   "<nom de l'entreprise tel qu'il apparaît sur le reçu, ou NOT FOUND>",
  "date":      "<date EXACTEMENT telle qu'elle apparaît sur le reçu, ex: 25/12/2018 ou 18-11-18, ou NOT FOUND>"
}
Règles strictes :
- Lire directement sur l'image, ne rien inventer.
- Pour "total" : prendre le montant final à payer (TOTAL, GRAND TOTAL, AMOUNT DUE).
- Pour "date" : recopier la date exactement comme sur le reçu, sans la convertir.
- Si une information n'est pas visible sur le reçu, écrire NOT FOUND.\
"""

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

Pour chaque champ, indique CORRECT, PARTIEL ou INCORRECT selon ces règles STRICTES et EXHAUSTIVES :

━━ Règle "total" ━━
Convertis les deux valeurs en nombre flottant en ignorant les symboles monétaires (RM, $, etc.).
  CORRECT  si les valeurs numériques sont strictement égales (ex: 80.9 == 80.90).
  PARTIEL  si l'écart relatif est ≤ 5 % (ex: réf=52.45, extrait=53.00 → écart=1.1% → PARTIEL).
  INCORRECT si l'écart > 5 %, ou si la valeur extraite est NOT FOUND.

━━ Règle "company" ━━
Trois cas possibles, dans l'ordre de priorité :
  CORRECT   : l'extrait contient EXACTEMENT les mêmes mots que la référence,
              ni plus ni moins (casse ignorée, ponctuation ignorée).
              Ex: réf="OJC MARKETING SDN BHD", extrait="OJC MARKETING SDN BHD" → CORRECT.
  PARTIEL   : (a) l'extrait contient tous les mots de la référence PLUS des mots supplémentaires
                  (ex: réf="OJC MARKETING SDN BHD", extrait="OJC MARKETING SDN BHD ROC" → PARTIEL),
              (b) l'extrait contient les mots principaux mais est incomplet
                  (ex: réf="HON HWA HARDWARE TRADING", extrait="HARDWARE TRADING" → PARTIEL),
              (c) le nom de marque visible sur le reçu est correct mais la raison sociale légale est absente
                  (ex: réf="GERBANG ALAF RESTAURANTS SDN BHD", extrait="McDonald's" → INCORRECT car nom différent).
  INCORRECT : faute d'orthographe sur un mot-clé du nom
              (ex: réf="RESTORAN WAN SHENG", extrait="RESTORAN HAN" → INCORRECT),
              ou nom complètement différent / inventé,
              ou valeur extraite est NOT FOUND.

━━ Règle "date" ━━
Convertis les deux dates en valeurs numériques (jour, mois, année) en reconnaissant tous les formats :
  DD/MM/YYYY, DD-MM-YY, DD MMM YY, YYYY-MM-DD, D/M/YY, etc.
  Les noms de mois anglais : Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6,
                             Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12.
  CORRECT   : jour + mois + année correspondent exactement
              (ex: "22 Mar 18" == "22/03/2018" == "22-03-18" → CORRECT).
  PARTIEL   : la date extraite correspond à ±1 jour calendaire
              (ex: réf="29/01/2018", extrait="28/01/2018" → écart=1 jour → PARTIEL).
  INCORRECT : écart > 1 jour, ou année incorrecte, ou NOT FOUND.
  Note : une heure en plus de la date correcte ne change pas le verdict
         (ex: "15/01/2019 11:05:16 AM" pour réf="15/01/2019" → CORRECT).

Réponds UNIQUEMENT avec un objet JSON valide :
{{
  "total":   {{"verdict": "<CORRECT|PARTIEL|INCORRECT>", "reason": "<explication courte>"}},
  "company": {{"verdict": "<CORRECT|PARTIEL|INCORRECT>", "reason": "<explication courte>"}},
  "date":    {{"verdict": "<CORRECT|PARTIEL|INCORRECT>", "reason": "<explication courte>"}}
}}\
"""


# ── Clients Ollama ────────────────────────────────────────────────────────────

def _ollama_client(host: str):
    try:
        import ollama
    except ImportError as exc:
        raise ImportError("pip install ollama") from exc
    return ollama.Client(host=host)


def _image_to_b64(img: Image.Image) -> str:
    """Encode une PIL Image en base64 PNG (format attendu par Ollama)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_info_phase1(host: str, model: str, ocr_text: str) -> dict:
    """Phase 1 : extraction depuis le texte OCR (Mistral ou équivalent)."""
    client = _ollama_client(host)
    prompt = EXTRACT_PROMPT_TEXT.format(ocr_text=ocr_text[:3000])
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0, "num_predict": 300},
    )
    raw = resp["message"]["content"].strip()
    try:
        data = json.loads(raw)
        return {
            "total":   str(data.get("total",   "NOT FOUND")).strip() or "NOT FOUND",
            "company": str(data.get("company", "NOT FOUND")).strip() or "NOT FOUND",
            "date":    str(data.get("date",    "NOT FOUND")).strip() or "NOT FOUND",
        }
    except json.JSONDecodeError:
        return {"total": "NOT FOUND", "company": "NOT FOUND", "date": "NOT FOUND"}


def extract_info_phase2(host: str, model: str, img: Image.Image) -> dict:
    """Phase 2 : extraction directement depuis l'image (Qwen2.5-VL ou équivalent).

    Pas de Tesseract — le modèle vision lit l'image brute du reçu.
    """
    client = _ollama_client(host)
    encoded = _image_to_b64(img)
    resp = client.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": EXTRACT_PROMPT_VISION,
            "images": [encoded],
        }],
        format="json",
        options={"temperature": 0.0, "num_predict": 300},
    )
    raw = resp["message"]["content"].strip()
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
               extracted: dict, reference: dict,
               fields: list[str]) -> dict:
    """LLM juge uniquement les champs listés dans `fields`.

    Le prompt ne mentionne que les champs à juger — les champs AUTO=CORRECT
    ne sont jamais soumis au LLM.

    Retourne un dict {field: verdict} uniquement pour les champs demandés.
    """
    if not fields:
        return {}

    client = _ollama_client(host)

    # Construire les lignes de référence et d'extrait uniquement pour les
    # champs à juger (les autres sont masqués pour ne pas perturber le LLM).
    ref_lines = "\n".join(
        f"  {f:<8} : {reference.get(f, '?')}" for f in fields
    )
    ext_lines = "\n".join(
        f"  {f:<8} : {extracted[f]}" for f in fields
    )
    fields_str = ", ".join(f.upper() for f in fields)

    prompt = JUDGE_PROMPT.format(
        ref_total=reference.get("total",   "?"),
        ref_company=reference.get("company", "?"),
        ref_date=reference.get("date",    "?"),
        ext_total=extracted["total"],
        ext_company=extracted["company"],
        ext_date=extracted["date"],
    )

    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0, "num_predict": 400},
    )
    raw = resp["message"]["content"].strip()
    try:
        data = json.loads(raw)
        return {
            field: data[field]["verdict"].upper()
            for field in fields          # uniquement les champs demandés
            if field in data and "verdict" in data[field]
        }
    except (json.JSONDecodeError, KeyError, AttributeError):
        return {}


# Ordre de sévérité des verdicts (du plus favorable au moins favorable)
_VERDICT_RANK = {"CORRECT": 3, "PARTIEL": 2, "NOT FOUND": 1, "INCORRECT": 0}


def merge_verdicts(auto: str, llm: str | None) -> str:
    """Fusionne AUTO et LLM en garantissant que le LLM ne peut pas dégrader AUTO.

    Règles :
      - AUTO == CORRECT   → FINAL = CORRECT   (LLM ignoré)
      - AUTO == NOT FOUND → FINAL = NOT FOUND (LLM ignoré — rien à juger)
      - Sinon             → FINAL = max(AUTO, LLM) selon l'ordre de sévérité
                            (on garde le verdict le plus favorable des deux)
    """
    if auto in ("CORRECT", "NOT FOUND"):
        return auto
    if llm is None:
        return auto
    return auto if _VERDICT_RANK.get(auto, 0) >= _VERDICT_RANK.get(llm, 0) else llm


# ── Comparaison automatique (sans LLM juge) ──────────────────────────────────

def _parse_amount(s: str) -> float | None:
    """Extrait la valeur flottante d'une chaîne monétaire (ignore RM, $, etc.)."""
    nums = re.findall(r"\d+[.,]\d+|\d+", s.replace(",", "."))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date(s: str):
    """
    Tente de parser une date sous forme (jour, mois, année) depuis tout format courant.
    Retourne (day, month, year) ou None si échec.
    Formats reconnus : DD/MM/YYYY, DD-MM-YY, YYYY-MM-DD, DD MMM YY, D/M/YY, etc.
    """
    s = s.strip()
    # Supprimer la partie heure si présente (ex: "15/01/2019 11:05:16 AM")
    s = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?(\s*(AM|PM))?$", "", s, flags=re.IGNORECASE).strip()

    # Normaliser les séparateurs
    # Essai "DD MMM YY" / "DD MMM YYYY" (ex: "22 Mar 18", "29 Jun 18")
    m = re.match(r"(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{2,4})$", s)
    if m:
        d, mon_str, y = int(m.group(1)), m.group(2).lower()[:3], int(m.group(3))
        mon = _MONTH_ABBR.get(mon_str)
        if mon:
            year = 2000 + y if y < 100 else y
            return (d, mon, year)

    # Essai YYYY-MM-DD
    m = re.match(r"(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})$", s)
    if m:
        return (int(m.group(3)), int(m.group(2)), int(m.group(1)))

    # Essai DD[-/.]MM[-/.]YYYY ou DD[-/.]MM[-/.]YY
    m = re.match(r"(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2,4})$", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 2000 + y if y < 100 else y
        return (d, mo, year)

    return None


def _date_diff_days(a: str, b: str) -> int | None:
    """Retourne la différence en jours entre deux chaînes de dates, ou None si parse échoue."""
    pa, pb = _parse_date(a), _parse_date(b)
    if pa is None or pb is None:
        return None
    try:
        from datetime import date
        da = date(pa[2], pa[1], pa[0])
        db = date(pb[2], pb[1], pb[0])
        return abs((da - db).days)
    except ValueError:
        return None


def _norm_company_words(s: str) -> set[str]:
    """Retourne l'ensemble des mots significatifs du nom (casse et ponctuation ignorées)."""
    return set(re.sub(r"[^\w\s]", " ", s.lower()).split())


def compare_field(field: str, extracted: str, reference: str) -> str:
    """
    Comparaison automatique par règle déterministe.

    TOTAL  : CORRECT si égal, PARTIEL si écart ≤ 5%, INCORRECT sinon.
    DATE   : CORRECT si même jour/mois/année (format-agnostique),
             PARTIEL si écart ≤ 1 jour, INCORRECT sinon.
    COMPANY: CORRECT si mots identiques exactement,
             PARTIEL si extrait ⊇ référence (surplus) ou ⊂ référence (incomplet mais clé),
             INCORRECT sinon (faute, nom différent).
    """
    if extracted.upper() == "NOT FOUND":
        return "NOT FOUND"

    # ── TOTAL ──────────────────────────────────────────────────────────────────
    if field == "total":
        ref_v = _parse_amount(reference)
        ext_v = _parse_amount(extracted)
        if ref_v is None or ext_v is None:
            return "INCORRECT"
        if ref_v == ext_v:
            return "CORRECT"
        if ref_v != 0 and abs(ext_v - ref_v) / abs(ref_v) <= 0.05:
            return "PARTIEL"
        return "INCORRECT"

    # ── DATE ───────────────────────────────────────────────────────────────────
    if field == "date":
        diff = _date_diff_days(extracted, reference)
        if diff is None:
            # Fallback : comparaison caractère-par-caractère normalisée
            def _strip(t):
                return re.sub(r"[\s\-/\.]", "", t).lower()
            return "CORRECT" if _strip(extracted) == _strip(reference) else "INCORRECT"
        if diff == 0:
            return "CORRECT"
        if diff == 1:
            return "PARTIEL"
        return "INCORRECT"

    # ── COMPANY ────────────────────────────────────────────────────────────────
    if field == "company":
        ref_words = _norm_company_words(reference)
        ext_words = _norm_company_words(extracted)
        if not ref_words:
            return "INCORRECT"
        if ref_words == ext_words:
            return "CORRECT"
        # Extrait contient tous les mots de la référence + surplus → PARTIEL
        if ref_words.issubset(ext_words):
            return "PARTIEL"
        # Extrait est un sous-ensemble de la référence (incomplet) :
        # PARTIEL si les mots communs couvrent ≥ 50% de la référence
        overlap = ref_words & ext_words
        if ref_words and len(overlap) / len(ref_words) >= 0.5:
            return "PARTIEL"
        return "INCORRECT"

    return "INCORRECT"


# ── Détection des triplets dataset ────────────────────────────────────────────

def find_triplets(dataset_dir: Path, split: str) -> list[tuple]:
    """
    Parcourt <dataset_dir>/<split>/ et retourne des triplets :
      (img_path, entities_dict, box_path | None)
    """
    split_dir = dataset_dir / split
    img_dir   = split_dir / "img"
    ent_dir   = split_dir / "entities"
    box_dir   = split_dir / "box"

    if not img_dir.exists():
        sys.exit(f"[ERROR] img/ introuvable : {img_dir}")
    if not ent_dir.exists():
        sys.exit(f"[ERROR] entities/ introuvable : {ent_dir}")

    triplets = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        ent_path = ent_dir / (img_path.stem + ".txt")
        if not ent_path.exists():
            continue
        try:
            raw = ent_path.read_bytes().decode("utf-8", errors="replace")
            raw = raw.replace("\r\n", "\n").replace("\r", "\n")
            entities = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON invalide ({ent_path.name}) : {exc}")
            continue

        box_path = (box_dir / (img_path.stem + ".txt")) if box_dir.exists() else None
        if box_path and not box_path.exists():
            box_path = None

        triplets.append((img_path, entities, box_path))

    return triplets


# ── Affichage ─────────────────────────────────────────────────────────────────

def print_doc_result(name, extracted, reference, auto, llm_verdicts, final):
    sep = "─" * 78
    print(f"\n{sep}")
    print(f"  {name}")
    print(f"  {'CHAMP':<10} {'RÉFÉRENCE':<26} {'EXTRAIT':<26} {'AUTO':<12} "
          f"{'LLM':<12} FINAL")
    print(f"  {'-'*10} {'-'*26} {'-'*26} {'-'*12} {'-'*12} {'-'*9}")
    for field in ("total", "company", "date"):
        ref_v  = str(reference.get(field, "?"))[:25]
        ext_v  = str(extracted.get(field, "?"))[:25]
        auto_v = auto.get(field, "?")
        llm_v  = llm_verdicts.get(field, "—")
        fin_v  = final.get(field, "?")
        print(f"  {field.upper():<10} {ref_v:<26} {ext_v:<26} "
              f"{color(auto_v, auto_v):<12} {color(llm_v, llm_v):<12} "
              f"{color(fin_v, fin_v)}")


def print_stats(results: list, pipeline: str) -> dict:
    n = len(results)
    if not n:
        return {}
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  RÉSULTATS GLOBAUX  ({n} reçus)  —  pipeline={pipeline.upper()}")
    print(sep)

    field_counts = {}
    for field in ("total", "company", "date"):
        counts = {"CORRECT": 0, "PARTIEL": 0, "INCORRECT": 0, "NOT FOUND": 0}
        for r in results:
            v = r.get("comparisons", {}).get(field, "INCORRECT")
            counts[v] = counts.get(v, 0) + 1
        field_counts[field] = counts
        cr = counts["CORRECT"]   / n * 100
        pr = counts["PARTIEL"]   / n * 100
        ir = counts["INCORRECT"] / n * 100
        nr = counts["NOT FOUND"] / n * 100
        print(f"\n  {field.upper()}")
        print(f"    Correct   : " + color(f"{counts['CORRECT']:3d} ({cr:5.1f}%)", "CORRECT"))
        print(f"    Partiel   : " + color(f"{counts['PARTIEL']:3d} ({pr:5.1f}%)", "PARTIEL"))
        print(f"    Incorrect : " + color(f"{counts['INCORRECT']:3d} ({ir:5.1f}%)", "INCORRECT"))
        print(f"    Non trouvé: {counts['NOT FOUND']:3d} ({nr:5.1f}%)")

    # Nombre de reçus dont les 3 champs sont CORRECT
    fully_correct = sum(
        all(r.get("comparisons", {}).get(f, "") == "CORRECT"
            for f in ("total", "company", "date"))
        for r in results
    )
    print(f"\n  ── Précision globale ──")
    print(f"  Reçus entièrement corrects (3/3) : "
          + color(f"{fully_correct}/{n}  ({fully_correct/n*100:.1f}%)", "CORRECT"))

    score = sum(
        (r.get("comparisons", {}).get("total",   "") == "CORRECT") +
        (r.get("comparisons", {}).get("company", "") == "CORRECT") +
        (r.get("comparisons", {}).get("date",    "") == "CORRECT") +
        0.5 * (r.get("comparisons", {}).get("total",   "") == "PARTIEL") +
        0.5 * (r.get("comparisons", {}).get("company", "") == "PARTIEL") +
        0.5 * (r.get("comparisons", {}).get("date",    "") == "PARTIEL")
        for r in results
    )
    print(f"  Score pondéré                    : {score:.1f} / {n * 3}"
          f"  ({score / (n * 3) * 100:.1f}%)")
    print(sep)

    return {
        "pipeline": pipeline,
        "n_receipts": n,
        "fully_correct": fully_correct,
        "fully_correct_pct": round(fully_correct / n * 100, 2),
        "weighted_score": score,
        "max_score": n * 3,
        "weighted_pct": round(score / (n * 3) * 100, 2),
        "per_field": field_counts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DocVQA  (Task-1 single-page + Task-3 infographics)
# ══════════════════════════════════════════════════════════════════════════════
#
# Contrairement a SROIE (3 champs fixes + entites JSON par image), DocVQA pose
# des questions libres avec une liste de reponses acceptees, dans un JSON unique
# par split. On reutilise l'OCR, le client Ollama et la distance de Levenshtein
# deja definis plus haut ; seuls le chargement, le prompt et le scoring changent.

# Nom de dossier de chaque tache dans la racine DocVQA.
DOCVQA_TASK_DIRS = {
    "task1": "Task-1_Single-Page-Document-Visual-Question-Answering",
    "task3": "Task-3_Infographics-VQA",
}


def _resolve_docvqa_annotation(task_dir: Path, task: str, split: str) -> Path | None:
    """Retourne le fichier d'annotations du split, en testant les noms connus."""
    ann_dir = task_dir / "Annotations"
    if task == "task1":
        candidates = [f"{split}_v1.0_withQT.json", f"{split}_v1.0.json"]
    else:  # task3
        candidates = [f"infographicsVQA_{split}_v1.0_withQT.json",
                      f"infographicsVQA_{split}_v1.0.json"]
    for name in candidates:
        if (ann_dir / name).exists():
            return ann_dir / name
    # Repli : premier JSON dont le nom contient le split.
    hits = sorted(ann_dir.glob(f"*{split}*.json")) if ann_dir.exists() else []
    return hits[0] if hits else None


def read_docvqa_ocr(ocr_path: Path) -> str:
    """
    Lit un OCR DocVQA fourni et retourne le texte ligne par ligne.

    Deux formats coexistent :
      - Task-1 : Microsoft Read  -> recognitionResults[].lines[].text
      - Task-3 : Amazon Textract -> LINE[].Text
    """
    try:
        d = json.loads(ocr_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return ""
    lines = []
    if isinstance(d, dict) and "recognitionResults" in d:        # Task-1
        for page in d["recognitionResults"]:
            for ln in page.get("lines", []):
                t = (ln.get("text") or "").strip()
                if t:
                    lines.append(t)
    elif isinstance(d, dict) and "LINE" in d:                    # Task-3
        for ln in d["LINE"]:
            t = (ln.get("Text") or "").strip()
            if t:
                lines.append(t)
    return "\n".join(lines)


def find_docvqa_items(dataset_dir: Path, tasks: list, split: str) -> list[dict]:
    """
    Parcourt les taches demandees et retourne une liste d'items :
      {task, questionId, question, answers, image_path, ocr_path | None}

    Seuls les items dont l'image existe reellement sont conserves (le full
    dataset reference des milliers d'images ; un echantillon n'en a qu'une
    partie). Les items sans cle "answers" (split test) sont ignores.
    """
    items = []
    for task in tasks:
        sub = DOCVQA_TASK_DIRS.get(task)
        task_dir = dataset_dir / sub if sub else None
        if not task_dir or not task_dir.exists():
            print(f"  [WARN] tache '{task}' introuvable : {task_dir}")
            continue

        ann_path = _resolve_docvqa_annotation(task_dir, task, split)
        if not ann_path:
            print(f"  [WARN] annotations '{split}' introuvables pour {task}")
            continue

        data = json.loads(ann_path.read_text(encoding="utf-8", errors="replace"))
        data = data["data"] if isinstance(data, dict) else data
        img_dir, ocr_dir = task_dir / "Images", task_dir / "OCR"

        n_before, n_no_ans, n_no_img = len(items), 0, 0
        for it in data:
            if not it.get("answers"):
                n_no_ans += 1
                continue
            if task == "task1":
                img_name = Path(it["image"]).name
                ocr_name = Path(img_name).stem + ".json"
            else:
                img_name = it["image_local_name"]
                ocr_name = it.get("ocr_output_file") or (Path(img_name).stem + ".json")

            img_path = img_dir / img_name
            if not img_path.exists():
                n_no_img += 1
                continue
            ocr_path = ocr_dir / ocr_name
            items.append({
                "task": task,
                "questionId": it.get("questionId"),
                "question": it["question"],
                "answers": list(it["answers"]),
                "image_path": img_path,
                "ocr_path": ocr_path if ocr_path.exists() else None,
            })

        kept = len(items) - n_before
        note = f" ({n_no_ans} sans reponse)" if n_no_ans else ""
        print(f"  {task}: {kept} item(s) retenu(s) sur {len(data)}"
              f" — {ann_path.name}{note}")

    return items


# ── ANLS (metrique officielle DocVQA) ─────────────────────────────────────────

def anls_nls(prediction: str, answers: list) -> float:
    """
    Score NLS brut (1 - distance normalisee minimale sur les reponses de ref),
    avant seuillage. Reproduit le pre-traitement officiel : minuscule, espaces
    normalises pour la distance ; longueur calculee sur les chaines brutes.
    """
    if not answers:
        return 0.0
    pred_norm = " ".join(prediction.strip().lower().split())
    values = []
    for ans in answers:
        ans_norm = " ".join(ans.strip().lower().split())
        dist = edit_distance(ans_norm, pred_norm)
        length = max(len(ans.upper()), len(prediction.upper()))
        values.append(0.0 if length == 0 else dist / length)
    return 1.0 - min(values)


def verdict_from_anls(nls: float, threshold: float = 0.5) -> str:
    """Verdict deterministe a partir du NLS brut (avant juge LLM)."""
    if nls >= 0.99:
        return "CORRECT"
    if nls >= threshold:
        return "PARTIEL"
    return "INCORRECT"


# ── Extraction Q/R libre (Phase 1 texte / Phase 2 vision) ─────────────────────

def _clean_answer(raw: str) -> str:
    a = raw.strip().strip('"').strip("'").strip()
    # Garder la premiere ligne non vide (les modeles ajoutent parfois du bla-bla).
    for line in a.splitlines():
        if line.strip():
            return line.strip()
    return a


def answer_phase1(host: str, model: str, ocr_text: str, question: str) -> str:
    """Phase 1 : repond a la question a partir du texte OCR."""
    client = _ollama_client(host)
    prompt = (
        "Tu es un assistant de question-reponse sur documents.\n"
        "Voici le texte extrait par OCR d'un document :\n\n"
        f"{ocr_text[:3000]}\n\n"
        f"Question : {question}\n\n"
        "Reponds UNIQUEMENT par la reponse exacte (un mot ou une courte "
        "expression telle qu'elle apparait dans le document), sans phrase. "
        "Si l'information est absente, reponds : NOT FOUND."
    )
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 100},
    )
    return _clean_answer(resp["message"]["content"])


def answer_phase2(host: str, model: str, img: Image.Image, question: str) -> str:
    """Phase 2 : repond a la question directement depuis l'image (vision)."""
    client = _ollama_client(host)
    prompt = (
        f"Question : {question}\n\n"
        "Lis le document de l'image et reponds UNIQUEMENT par la reponse "
        "exacte (un mot ou une courte expression), sans phrase. "
        "Si l'information est absente, reponds : NOT FOUND."
    )
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt,
                   "images": [_image_to_b64(img)]}],
        options={"temperature": 0.0, "num_predict": 100},
    )
    return _clean_answer(resp["message"]["content"])


def judge_docvqa(host: str, model: str, question: str,
                 answers: list, prediction: str) -> dict:
    """Juge LLM : la prediction repond-elle correctement a la question ?"""
    client = _ollama_client(host)
    refs = " | ".join(answers)
    prompt = (
        "Tu es un evaluateur strict de question-reponse sur documents.\n"
        f"Question : {question}\n"
        f"Reponses de reference acceptees : {refs}\n"
        f"Reponse du modele : {prediction}\n\n"
        "Rends un objet JSON, sans texte autour :\n"
        '{"verdict": "<CORRECT|PARTIEL|INCORRECT>", "reason": "<courte explication>"}\n'
        "- CORRECT : equivalent a une reference (casse/espaces/format sans importance).\n"
        "- PARTIEL : recouvre partiellement la bonne reponse (incomplet ou en surplus).\n"
        "- INCORRECT : reponse fausse ou hors-sujet."
    )
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0, "num_predict": 150},
        )
        data = json.loads(resp["message"]["content"].strip())
        verdict = str(data.get("verdict", "")).upper().strip()
        if verdict not in ("CORRECT", "PARTIEL", "INCORRECT"):
            verdict = "INCORRECT"
        return {"verdict": verdict, "reason": str(data.get("reason", "")).strip()}
    except (json.JSONDecodeError, KeyError, Exception):
        return {"verdict": "INCORRECT", "reason": "juge indisponible"}


# ── Stats DocVQA ──────────────────────────────────────────────────────────────

def print_stats_docvqa(results: list, pipeline: str, threshold: float) -> dict:
    n = len(results)
    if not n:
        return {}
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  RESULTATS DocVQA  ({n} questions)  —  pipeline={pipeline.upper()}")
    print(sep)

    # ANLS officiel : NLS seuille puis moyenne.
    anls = sum((r["nls"] if r["nls"] >= threshold else 0.0) for r in results) / n

    counts = {"CORRECT": 0, "PARTIEL": 0, "INCORRECT": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    accuracy = counts["CORRECT"] / n
    weighted = (counts["CORRECT"] + 0.5 * counts["PARTIEL"]) / n

    print(f"\n  ANLS (seuil {threshold})            : "
          + color(f"{anls*100:5.2f}%", "CORRECT"))
    print(f"  Exactes (CORRECT)            : "
          + color(f"{counts['CORRECT']:3d} ({accuracy*100:5.1f}%)", "CORRECT"))
    print(f"  Partielles                   : "
          + color(f"{counts['PARTIEL']:3d} ({counts['PARTIEL']/n*100:5.1f}%)", "PARTIEL"))
    print(f"  Incorrectes                  : "
          + color(f"{counts['INCORRECT']:3d} ({counts['INCORRECT']/n*100:5.1f}%)", "INCORRECT"))
    print(f"  Score pondere (C + 0.5*P)    : {weighted*100:5.1f}%")

    # Detail par tache (si plusieurs).
    per_task = {}
    tasks = sorted({r["task"] for r in results})
    if len(tasks) > 1:
        print(f"\n  ── Par tache ──")
        for t in tasks:
            sub = [r for r in results if r["task"] == t]
            a = sum((r["nls"] if r["nls"] >= threshold else 0.0) for r in sub) / len(sub)
            c = sum(r["verdict"] == "CORRECT" for r in sub)
            per_task[t] = {"n": len(sub), "anls_pct": round(a * 100, 2),
                           "correct": c, "accuracy_pct": round(c / len(sub) * 100, 2)}
            print(f"    {t:6s} : {len(sub):3d} Q — ANLS {a*100:5.2f}% — "
                  f"exactes {c}/{len(sub)} ({c/len(sub)*100:.1f}%)")
    print(sep)

    return {
        "pipeline": pipeline,
        "dataset": "docvqa",
        "n_questions": n,
        "anls_threshold": threshold,
        "anls_pct": round(anls * 100, 2),
        "accuracy_pct": round(accuracy * 100, 2),
        "weighted_pct": round(weighted * 100, 2),
        "verdict_counts": counts,
        "per_task": per_task,
    }


def run_docvqa(args):
    """Boucle d'evaluation DocVQA (appelee par main quand --dataset-type docvqa)."""
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset introuvable : {dataset_dir}")

    split = args.split or "val"          # test n'a pas de reponses publiques
    tasks = ["task1", "task3"] if args.task == "both" else [args.task]
    pipeline = args.pipeline

    items = find_docvqa_items(dataset_dir, tasks, split)
    if not items:
        sys.exit(f"[ERROR] Aucun item DocVQA (taches={tasks}, split={split}). "
                 f"Note : le split 'test' n'a pas de reponses publiques.")
    if args.max_docs:
        items = items[:args.max_docs]

    if pipeline == "phase1":
        ocr_label = "OCR fourni (OCR/)" if args.use_provided_ocr else ocr_engine_label(args)
        model_label = args.ollama_model
    else:
        ocr_label = "aucun (image directe)"
        model_label = args.vision_model

    print(f"\n  {len(items)} question(s)")
    print(f"  dataset      = DocVQA  (taches={'+'.join(tasks)})")
    print(f"  pipeline     = {pipeline.upper()}")
    print(f"  OCR          = {ocr_label}")
    print(f"  modele       = {model_label}")
    print(f"  split        = {split}")

    results = []
    for i, it in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] [{it['task']}] {it['image_path'].name}")
        print(f"  Q: {it['question']}")

        try:
            if pipeline == "phase1":
                if args.use_provided_ocr and it["ocr_path"]:
                    ocr_text = read_docvqa_ocr(it["ocr_path"])
                    src = "ocr_fourni"
                else:
                    img = load_image(it["image_path"], args.dpi)
                    ocr_text = run_ocr_text(img, args)
                    src = args.ocr_engine
                prediction = answer_phase1(args.ollama_host, args.ollama_model,
                                           ocr_text, it["question"])
            else:
                img = load_image(it["image_path"], args.dpi)
                prediction = answer_phase2(args.ollama_host, args.vision_model,
                                           img, it["question"])
                src = "none (vision directe)"
        except Exception as exc:
            print(f"  [WARN] item ignore : {exc}")
            continue

        nls = anls_nls(prediction, it["answers"])
        auto = verdict_from_anls(nls, args.anls_threshold)
        if auto != "CORRECT":
            judged = judge_docvqa(args.ollama_host, args.ollama_model,
                                  it["question"], it["answers"], prediction)
        else:
            judged = {"verdict": "CORRECT", "reason": ""}
        verdict = merge_verdicts(auto, judged.get("verdict"))

        print(f"  → reponse={prediction!r}  | ref={it['answers']}")
        print(f"    NLS={nls:.3f}  auto={color(auto, auto)}  "
              f"juge={color(judged['verdict'], judged['verdict'])}  "
              f"final={color(verdict, verdict)}")

        results.append({
            "task": it["task"],
            "questionId": it["questionId"],
            "file": it["image_path"].name,
            "question": it["question"],
            "answers": it["answers"],
            "prediction": prediction,
            "ocr_source": src,
            "nls": nls,
            "auto_verdict": auto,
            "llm_verdict": judged.get("verdict"),
            "verdict": verdict,
        })

    if results:
        summary = print_stats_docvqa(results, pipeline, args.anls_threshold)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": results},
                          f, ensure_ascii=False, indent=2)
            print(f"\n  Resultats sauvegardes → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Évaluation par lot Phase 1 / Phase 2 sur SROIE (defaut) "
                    "ou DocVQA (--dataset-type docvqa)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pipeline",  default="phase1",
                        choices=["phase1", "phase2"],
                        help="phase1 = Tesseract + LLM texte (Mistral)  "
                             "phase2 = image → LLM vision (Qwen2.5-VL), pas de Tesseract")
    parser.add_argument("--dataset-type", default="sroie",
                        choices=["sroie", "docvqa"],
                        help="Type de dataset : sroie (defaut) ou docvqa")
    parser.add_argument("--task", default="both",
                        choices=["task1", "task3", "both"],
                        help="DocVQA uniquement : task1 (single-page), "
                             "task3 (infographics) ou both (defaut)")
    parser.add_argument("--dataset",   required=True,
                        help="Racine du dataset (contient test/, train/, ...)")
    parser.add_argument("--split",     default=None,
                        help="Sous-dossier/split a evaluer "
                             "(SROIE: defaut test — DocVQA: defaut val)")
    parser.add_argument("--out",       default=None,
                        help="Fichier JSON résultats (optionnel)")
    parser.add_argument("--lang",      default="fra+eng",
                        help="Codes langue Tesseract — Phase 1 uniquement (défaut: fra+eng)")
    parser.add_argument("--dpi",       type=int, default=300,
                        help="DPI de rasterisation des PDFs (défaut: 300)")
    parser.add_argument("--max-docs",  type=int, default=None,
                        help="Limite le nombre de documents traités")
    parser.add_argument("--enhance",   default="auto",
                        choices=["auto", "on", "off"],
                        help="Filtre contraste Tesseract — Phase 1 uniquement")
    parser.add_argument("--alpha",     type=float, default=1.5)
    parser.add_argument("--beta",      type=int,   default=0)
    parser.add_argument("--use-box-annotations", action="store_true",
                        help="Phase 1 : utiliser les annotations box/ au lieu de Tesseract")
    parser.add_argument("--use-provided-ocr", action="store_true",
                        help="DocVQA Phase 1 : utiliser le dossier OCR/ fourni "
                             "au lieu de lancer Tesseract")
    parser.add_argument("--anls-threshold", type=float, default=0.5,
                        help="Seuil ANLS DocVQA (defaut: 0.5)")
    parser.add_argument("--ocr-engine", default=None,
                        choices=["doctr", "tesseract"],
                        help="Moteur OCR Phase 1 (defaut: celui du projet, "
                             "soit docTR). Les options --lang/--enhance/--alpha/"
                             "--beta ne s'appliquent qu'a tesseract.")
    parser.add_argument("--ollama-host",    default=os.environ.get(
                        "DOQMENT_OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--ollama-model",   default=os.environ.get(
                        "DOQMENT_OLLAMA_TEXT_MODEL", "mistral:7b-instruct"),
                        help="Modèle texte Ollama — Phase 1 extraction + juge (défaut: mistral:7b-instruct)")
    parser.add_argument("--vision-model",   default=os.environ.get(
                        "DOQMENT_OLLAMA_VISION_MODEL", "qwen2.5vl:7b"),
                        help="Modèle vision Ollama — Phase 2 extraction (défaut: qwen2.5vl:7b)")
    args = parser.parse_args()

    if args.ocr_engine is None:
        args.ocr_engine = _project_default_ocr_engine()

    if args.dataset_type == "docvqa":
        return run_docvqa(args)

    split = args.split or "test"
    pipeline = args.pipeline

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset introuvable : {dataset_dir}")

    triplets = find_triplets(dataset_dir, split)
    if not triplets:
        sys.exit(f"[ERROR] Aucun triplet img+entities trouvé dans "
                 f"{dataset_dir / split}")

    if args.max_docs:
        triplets = triplets[:args.max_docs]

    # Résumé de configuration
    if pipeline == "phase1":
        ocr_label = "annotations box/" if args.use_box_annotations else ocr_engine_label(args)
        model_label = args.ollama_model
    else:
        ocr_label = "aucun (image directe)"
        model_label = args.vision_model

    print(f"\n  {len(triplets)} reçu(s)")
    print(f"  pipeline     = {pipeline.upper()}")
    print(f"  OCR          = {ocr_label}")
    print(f"  modèle       = {model_label}")
    print(f"  split        = {split}")

    results = []

    for i, (img_path, reference, box_path) in enumerate(triplets, 1):
        print(f"\n[{i}/{len(triplets)}] {img_path.name}")

        record = {
            "file": img_path.name,
            "pipeline": pipeline,
            "reference": reference,
        }

        # ── Phase 1 : Tesseract → LLM texte ──────────────────────────────────
        if pipeline == "phase1":
            if args.use_box_annotations and box_path:
                ocr_text = read_box_annotation(box_path)
                print(f"  Annotation box/ : {len(ocr_text)} chars")
            else:
                try:
                    img = load_image(img_path, args.dpi)
                    ocr_text = run_ocr_text(img, args)
                    print(f"  {ocr_engine_label(args)} OCR : {len(ocr_text)} chars")
                except Exception as exc:
                    print(f"  [WARN] OCR échoué : {exc}")
                    continue

            extracted = extract_info_phase1(args.ollama_host, args.ollama_model,
                                            ocr_text)
            record["ocr_source"] = "box_annotations" if (args.use_box_annotations and box_path) \
                                   else args.ocr_engine

        # ── Phase 2 : image → LLM vision ─────────────────────────────────────
        else:
            try:
                img = load_image(img_path, args.dpi)
            except Exception as exc:
                print(f"  [WARN] Chargement image échoué : {exc}")
                continue

            print(f"  Vision ({args.vision_model}) ← {img_path.name} "
                  f"({img.width}×{img.height}px)")
            extracted = extract_info_phase2(args.ollama_host, args.vision_model, img)
            record["ocr_source"] = "none (vision directe)"

        print(f"  → company={extracted['company']!r}  "
              f"total={extracted['total']!r}  date={extracted['date']!r}")

        # ── Comparaison (auto + juge LLM si nécessaire) ──────────────────────
        auto_comparisons = {
            f: compare_field(f, extracted[f], reference.get(f, ""))
            for f in ("total", "company", "date")
        }

        # Le LLM n'est consulté que pour les champs PARTIEL ou INCORRECT.
        # AUTO=CORRECT est certain — figé immédiatement.
        # AUTO=NOT FOUND est certain aussi — le LLM ne peut rien extraire
        # de plus que ce que le modèle d'extraction a déjà raté.
        fields_needing_llm = [
            f for f in ("total", "company", "date")
            if auto_comparisons[f] not in ("CORRECT", "NOT FOUND")
        ]

        if fields_needing_llm:
            llm_verdicts = judge_info(args.ollama_host, args.ollama_model,
                                      extracted, reference,
                                      fields=fields_needing_llm)
        else:
            llm_verdicts = {}  # tous CORRECT, pas besoin du LLM

        # Fusion : AUTO=CORRECT est figé ; pour les autres on prend le
        # verdict le plus favorable entre AUTO et LLM (le LLM ne peut
        # pas dégrader un AUTO=PARTIEL en INCORRECT).
        comparisons = {
            f: merge_verdicts(auto_comparisons[f], llm_verdicts.get(f))
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
        results.append(record)

    # ── Stats globales ────────────────────────────────────────────────────────
    if results:
        summary = print_stats(results, pipeline)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            output = {"summary": summary, "results": results}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"\n  Résultats sauvegardés → {out_path}")


if __name__ == "__main__":
    main()
