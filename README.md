# TrAitement-DoQment

Système de gestion documentaire local : on dépose des PDF ou images, on pose une question en langage naturel, on obtient une réponse sourcée pointant la zone exacte du document. Tout tourne sur la machine de l'utilisateur — aucune donnée ne quitte la machine. Conforme à l'esprit de la Loi 25 du Québec.

> *Atelier en intelligence artificielle — BEHAREL Armand, JEANNE Arthur, SOUKI Mohamed*

---

## Architecture

**Deux pipelines indépendants**, **trois commandes chacun**, **un seul backend LLM** (Ollama).

|  | Pipeline 1 — *textuel* | Pipeline 2 — *multimodal* |
|---|---|---|
| Représentation | Texte OCR → embeddings MPNet 768d | Image de page → embeddings visuels ColQwen2 |
| Index | FAISS HNSW | Qdrant (local-path) |
| Génération | **Ollama** + `mistral:7b-instruct` | **Ollama** + `qwen2.5vl:7b` |
| OCR requis | Oui (**docTR** par défaut, ou Tesseract / annotations) | Non |

```
scripts/phase1.py {ingest, doc, db}    ← pipeline textuel
scripts/phase2.py {ingest, doc, db}    ← pipeline multimodal
```

- **`ingest`** construit l'index à partir d'un dossier de documents
- **`doc`** répond à une question sur **un seul** document, sans toucher à l'index
- **`db`** répond à une question contre **toute** la base indexée (réponses multi-documents)

L'interface Streamlit (`app.py`) expose les mêmes trois modes via un sélecteur dans la barre latérale.

---

## Démarrage rapide

```bash
# 1. Environnement Python.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Ollama et les modèles.
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct
ollama pull qwen2.5vl:7b           # ~5 Go, uniquement si Pipeline 2

# 3. OCR Pipeline 1 : docTR (moteur par défaut) est installé par requirements.txt.
#    Tesseract n'est requis que si vous forcez --ocr-engine tesseract :
sudo dnf install tesseract tesseract-langpack-fra tesseract-langpack-eng
# OU :  sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng

# 4. Tests (doivent afficher 61 passed).
pytest

# 5. Interface Streamlit.
streamlit run app.py
```

---

## Importer les jeux de données

Aucun téléchargement automatique.

**SROIE-Dataset_v2** (Pipeline 1 + évaluation) : décompressez sous `data/SROIE-Dataset_v2/` à la racine :

```
data/SROIE-Dataset_v2/
└── test/
    ├── img/        ← images .jpg des reçus
    ├── box/        ← annotations ICDAR (x1,y1,...,x4,y4,texte)
    └── entities/   ← JSON ground-truth {company, date, address, total}
```

**DocVQA** (évaluation Phase 1/2 — Task-1 single-page + Task-3 infographics) : décompressez sous `data/DocVQA/` :

```
data/DocVQA/
├── Task-1_Single-Page-Document-Visual-Question-Answering/
│   ├── Annotations/   ← <split>_v1.0_withQT.json (questions + réponses)
│   ├── Images/        ← .png
│   └── OCR/           ← .json (OCR fourni, optionnel)
└── Task-3_Infographics-VQA/
    ├── Annotations/   ← infographicsVQA_<split>_v1.0_withQT.json
    ├── Images/        ← .jpeg
    └── OCR/           ← .json
```
> Le split `test` n'a pas de réponses publiques → évaluer sur `val` (défaut) ou `train`.

**M3DocVQA** (Pipeline 2, benchmark optionnel) : <https://huggingface.co/datasets/JaehyeongJung/m3docvqa>

**Vos PDF / images** (Pipeline 2) : déposez-les dans `data/raw/`.

---

## Pipeline 1 — textuel

```bash
# Indexer SROIE-Dataset_v2 — pointer sur le dossier racine suffit.
python scripts/phase1.py ingest --dir data/SROIE-Dataset_v2

# Pour les images sans annotation box/, activer l'OCR (docTR par défaut) :
python scripts/phase1.py ingest --dir data/SROIE-Dataset_v2 --ocr

# Forcer Tesseract au lieu de docTR :
python scripts/phase1.py ingest --dir data/SROIE-Dataset_v2 --ocr --ocr-engine tesseract

# Question contre toute la base.
python scripts/phase1.py db --question "What is the total amount?"

# Question sur un seul document (avec ou sans annotation).
python scripts/phase1.py doc \
    --file "data/SROIE-Dataset_v2/test/img/X51005268420.jpg" \
    --question "What is the company name?" \
    --annotation "data/SROIE-Dataset_v2/test/box/X51005268420.txt"
```

L'annotation `--annotation` est facultative ; sans elle, l'OCR (docTR par défaut, sinon Tesseract via `--ocr-engine tesseract`) tourne automatiquement sur l'image.

---

## Pipeline 2 — multimodal

```bash
# Indexer un dossier de PDF / images (idempotent via MD5).
python scripts/phase2.py ingest --dir data/raw/

# Question contre toute la base.
python scripts/phase2.py db --question "Quel est le montant total de la facture Acme ?"

# Question sur un seul document.
python scripts/phase2.py doc --file path/to/invoice.pdf --question "..."
```

---

## Évaluation de la précision (SROIE + DocVQA)

Le script `scripts/ocr_eval_batch.py` mesure la précision des Phases 1 et 2 sur deux jeux de données, choisis via `--dataset-type` :

- **`sroie`** (défaut) — pour chaque reçu, trois questions fixes (company, total, date) comparées à la ground-truth `entities/`.
- **`docvqa`** — questions libres (Task-1 + Task-3) comparées aux réponses de référence ; métrique **ANLS** (standard DocVQA) en plus du verdict CORRECT/PARTIEL/INCORRECT.

L'OCR de la Phase 1 utilise **docTR par défaut** (comme le reste du projet) ; `--ocr-engine tesseract` rétablit Tesseract.

Le flag `--pipeline` choisit le modèle évalué :

| `--pipeline` | OCR | Modèle d'extraction | Modèle juge |
|---|---|---|---|
| `phase1` | **docTR** (défaut) ou Tesseract | `--ollama-model` (Mistral) | `--ollama-model` (Mistral) |
| `phase2` | **aucun** — image directe | `--vision-model` (Qwen2.5-VL) | `--ollama-model` (Mistral) |

```bash
# ════════════════════════ SROIE (défaut) ════════════════════════
# ── Phase 1 (docTR + Mistral) ──────────────────────────────────────────────
python scripts/ocr_eval_batch.py \
    --pipeline phase1 \
    --dataset  data/SROIE-Dataset_v2 \
    --split    test \
    --out      data/eval/results_phase1.json

# ── Phase 2 (image → Qwen2.5-VL, pas d'OCR) ────────────────────────────────
python scripts/ocr_eval_batch.py \
    --pipeline phase2 \
    --dataset  data/SROIE-Dataset_v2 \
    --split    test \
    --out      data/eval/results_phase2.json

# ── Test rapide (10 reçus) ───────────────────────────────────────────────────
python scripts/ocr_eval_batch.py \
    --pipeline phase1 \
    --dataset  data/SROIE-Dataset_v2 --split test \
    --max-docs 10

# ── Phase 1 avec annotations box/ (bypass OCR, évalue extraction seule) ──────
python scripts/ocr_eval_batch.py \
    --pipeline phase1 --use-box-annotations \
    --dataset  data/SROIE-Dataset_v2 --split test

# ── Forcer Tesseract au lieu de docTR ───────────────────────────────────────
python scripts/ocr_eval_batch.py \
    --pipeline phase1 --ocr-engine tesseract \
    --dataset  data/SROIE-Dataset_v2 --split test

# ════════════════════════ DocVQA ════════════════════════
# ── Phase 1, Task-1 + Task-3, split val (réponses publiques) ────────────────
python scripts/ocr_eval_batch.py \
    --dataset-type docvqa \
    --dataset      data/DocVQA \
    --task         both \
    --split        val \
    --pipeline     phase1 \
    --out          data/eval/results_docvqa_phase1.json

# ── Phase 2 (vision), une seule tâche ───────────────────────────────────────
python scripts/ocr_eval_batch.py \
    --dataset-type docvqa --dataset data/DocVQA \
    --task task1 --split val --pipeline phase2

# ── Phase 1 en réutilisant l'OCR/ fourni au lieu de docTR ───────────────────
python scripts/ocr_eval_batch.py \
    --dataset-type docvqa --dataset data/DocVQA \
    --task both --split val --pipeline phase1 --use-provided-ocr
```

**Métriques rapportées (SROIE)** :

| Métrique | Description |
|---|---|
| Reçus entièrement corrects | Reçus où les 3 champs (company, total, date) sont CORRECT |
| Score pondéré | CORRECT=1 pt, PARTIEL=0.5 pt, INCORRECT=0 pt — sur N×3 pts |
| Par champ | Taux CORRECT / PARTIEL / INCORRECT / NOT FOUND par field |

**Métriques rapportées (DocVQA)** :

| Métrique | Description |
|---|---|
| ANLS | Average Normalized Levenshtein Similarity (seuil `--anls-threshold`, défaut 0.5) — métrique standard DocVQA |
| Exactitude | Questions au verdict CORRECT / total |
| Score pondéré | CORRECT=1 pt, PARTIEL=0.5 pt — sur N pts |
| Par tâche | ANLS et exactitude détaillés pour task1 / task3 (si `--task both`) |

**Verdicts** (auto déterministe → affiné par le juge LLM, fusion la plus favorable) :
- `CORRECT` — réponse équivalente à la référence
- `PARTIEL` — partiellement correcte (SROIE company : ≥ 60 % des mots-clés ; DocVQA : ANLS ≥ seuil)
- `INCORRECT` — réponse incorrecte
- `NOT FOUND` — le modèle n'a pas trouvé l'information (SROIE)

> Le **juge** est le modèle texte (Mistral) consulté quand le verdict automatique n'est pas déjà CORRECT, pour rattraper les équivalences de sens que la distance de caractères rate (ex. `2` vs `two`). L'ANLS, lui, reste calculé indépendamment du juge.

---

## Configuration

Tout vit dans `doqment/settings.py` — une seule dataclass `Settings` avec des défauts sensés. Pour surcharger sans toucher au code, exporter des variables d'environnement :

```bash
export DOQMENT_OLLAMA_HOST=http://localhost:11434
export DOQMENT_OLLAMA_TEXT_MODEL=mistral:7b-instruct
export DOQMENT_OLLAMA_VISION_MODEL=qwen2.5vl:7b
export DOQMENT_OCR_ENGINE=doctr            # OCR Phase 1 : "doctr" (défaut) ou "tesseract"
export DOQMENT_COLQWEN_DEVICE=cuda:0       # ou "cpu"
export DOQMENT_COLQWEN_DTYPE=bfloat16
```

---

## Structure du dépôt

```
TrAitement-DoQment/
├── ingestion.py               ← canonique coéquipier 1 (intouché)
├── pipeline1.py               ← canonique coéquipier 1 (intouché)
├── pipeline_rag2.py           ← canonique coéquipier 2 (intouché)
├── pdf_ocr.py                 ← canonique coéquipier 2 (intouché)
├── Comparaison_OCR.py         ← benchmark OCR coéquipier 2 (adapté : docTR + Tesseract)
│
├── app.py                     ← interface Streamlit (3 vues)
├── conftest.py
├── requirements.txt
│
├── doqment/                   ← TOUT notre code, un seul package plat
│   ├── settings.py            ← config (Settings dataclass)
│   ├── llm.py                 ← client Ollama (texte + vision)
│   ├── ocr.py                 ← wrapper OCR (docTR par défaut + Tesseract)
│   ├── phase1.py              ← Pipeline 1 (ingest_directory, ask_document, ask_database)
│   ├── phase2.py              ← Pipeline 2 (idem, multimodal)
│   ├── phase2_store.py        ← ColQwen2 + Qdrant + SQLite + rasterize
│   └── ui/                    ← 3 vues Streamlit (doc, db, ingest)
│
├── scripts/
│   ├── phase1.py              ← CLI Pipeline 1
│   ├── phase2.py              ← CLI Pipeline 2
│   └── ocr_eval_batch.py      ← évaluation précision Phase 1/2 (SROIE + DocVQA)
│
├── tests/                     ← 61 tests utiles, exécutés en <3 secondes
│   ├── test_settings.py
│   ├── test_llm.py            ← parsing JSON du VLM, validation citations
│   ├── test_ocr.py            ← groupement mots→lignes Tesseract
│   ├── test_phase1.py         ← ask_document + ask_database avec mocks
│   └── test_phase2.py         ← idempotence MetadataStore, maxsim, rasterize
│
└── docs/
```

---

## Dépannage

**`tesseract is not installed or it's not in your PATH`** — N'arrive que si vous forcez `--ocr-engine tesseract` (ou `DOQMENT_OCR_ENGINE=tesseract`). L'OCR par défaut étant docTR (installé par `requirements.txt`), Pipeline 1 fonctionne sans le binaire Tesseract. Sinon, installez Tesseract (voir Démarrage rapide) ou fournissez l'annotation ICDAR jumelle.

**`Connection refused` quand le pipeline appelle Ollama** — Démarrer le démon : `ollama serve` (souvent automatique après l'installation). Vérifier les modèles tirés : `ollama list`.

**`[transformers] Accessing __path__ from ...`** — Pré-avertissement de dépréciation de la version récente de `transformers`. Aucun effet sur le code.

**HuggingFace : `404 Not Found` sur `adapter_config.json` / `preprocessor_config.json`** — Normal. `sentence-transformers` cherche des fichiers optionnels qui n'existent pas pour MPNet.

**`Could not load library with AVX2 support` (FAISS)** — Le fallback générique s'active. Fonctionnel, performance légèrement plus basse.

**`AttributeError: 'OCREngine' object has no attribute '_model'`** — Bug du fichier canonique `ingestion.py:82-99`. Le mode `doc` Pipeline 1 contourne automatiquement en passant par `doqment/ocr.py` (docTR par défaut, ou Tesseract). Si vous voyez l'erreur, c'est que vous tournez une version antérieure — `pytest` doit afficher 61 passed.

**`ResponseError: model requires more system memory (12.5 GiB) than is available (...)`** — Qwen2.5-VL 7B requiert ~12,5 Go de RAM en CPU-only (~5 Go en GPU CUDA). Libérer de la RAM (fermer le navigateur, IDE, etc.) ou exécuter sur une machine équipée. Le projet est figé sur ce modèle ; ne pas le remplacer.

**`Found no NVIDIA driver on your system`** sur Pipeline 2 — ColQwen2 cherche un GPU CUDA. Sur machine CPU-only :
```bash
export DOQMENT_COLQWEN_DEVICE=cpu
export DOQMENT_COLQWEN_DTYPE=float32
```
L'encodage est ~50× plus lent qu'avec GPU (compter ~30 s par page) — réaliste sur 1-10 documents seulement.

---

## Quatre fichiers canoniques cohabitent

Les coéquipiers ont livré du code Python qu'on conserve **byte-identique aux originaux** :

| Fichier | Apport | Qui s'en sert |
|---|---|---|
| `ingestion.py` + `pipeline1.py` | Indexation canonique SROIE (coéquipier 1) | `doqment/phase1.py` |
| `pipeline_rag2.py` | RAG complet avec REPL (coéquipier 2) | usage standalone |
| `pdf_ocr.py` | OCR Tesseract autonome (coéquipier 2) | extraction texte hors RAG |

`Comparaison_OCR.py` (benchmark OCR du coéquipier 2) a été **adapté** depuis : support docTR ajouté en plus de Tesseract, Surya/Kraken retirés. Il n'est donc plus byte-identique à l'original.

---

## Limites assumées

- **L'index FAISS HNSW** (Pipeline 1) ne supporte pas la suppression individuelle — il faut réindexer pour retirer un document.
- **`ingestion.py:32`** : `BoundingBox.ymax` contient un bug (`max(self.x1, self.y2, self.y3, self.y4)` au lieu de `self.y1`). Contourné dans nos tests.
- **`ingestion.py:82-99`** : `OCREngine._load()` référence `self._model` qui n'existe pas. Contourné par `doqment/ocr.py` (docTR par défaut, ou Tesseract).
- **Pipeline 2 demande de la RAM** — Qwen2.5-VL 7B en CPU-only consomme ~12,5 Go (≈ 5 Go avec GPU CUDA). ColQwen2 ajoute ~6 Go. Sur machine ≤ 16 Go libre, fermer les autres applications avant de lancer `db`. Ce dimensionnement est assumé.
- **La Loi 25** — l'exécution locale ne couvre que l'aspect *technique*. Le consentement, le registre et le RPRP restent à instrumenter au niveau organisationnel.
