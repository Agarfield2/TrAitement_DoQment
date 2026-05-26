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
| OCR requis | Oui (Tesseract ou annotations) | Non |

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

# 3. Tesseract (Pipeline 1 sans annotations SROIE).
sudo dnf install tesseract tesseract-langpack-fra tesseract-langpack-eng
# OU :  sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng

# 4. Tests (doivent afficher 63 passed).
pytest

# 5. Interface Streamlit.
streamlit run app.py
```

---

## Importer les jeux de données

Aucun téléchargement automatique.

**SROIE2019** (Pipeline 1) : <https://drive.google.com/drive/folders/1ShItNWXyiY1tFDM5W02bceHuJjyeeJl2>
Décompressez sous `data/SROIE2019/` à la racine :

```
data/SROIE2019/
├── 0325updated.task1train(626p)/    ← images .jpg + annotations ICDAR .txt
└── 0325updated.task2train(626p)/    ← entités JSON (company, date, total, address)
```

**M3DocVQA** (Pipeline 2, benchmark optionnel) : <https://huggingface.co/datasets/JaehyeongJung/m3docvqa>

**Vos PDF / images** (Pipeline 2) : déposez-les dans `data/raw/`.

---

## Pipeline 1 — textuel

```bash
# Indexer SROIE — pointer sur le dossier racine suffit, l'ingestion
# scanne récursivement et auto-détecte les fichiers d'entités JSON.
python scripts/phase1.py ingest --dir data/SROIE2019

# Pour les documents sans annotation SROIE, ajouter --tesseract
# (sinon ils sont silencieusement sautés) :
python scripts/phase1.py ingest --dir data/SROIE2019 --tesseract

# Question contre toute la base.
python scripts/phase1.py db --question "What is the total amount?"

# Question sur un seul document (avec ou sans annotation).
python scripts/phase1.py doc \
    --file "data/SROIE2019/0325updated.task1train(626p)/X51005268420.jpg" \
    --question "What is the company name?" \
    --annotation "data/SROIE2019/0325updated.task1train(626p)/X51005268420.txt"
```

L'annotation `--annotation` est facultative ; sans elle, Tesseract OCR tourne automatiquement sur l'image.

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

## Configuration

Tout vit dans `doqment/settings.py` — une seule dataclass `Settings` avec des défauts sensés. Pour surcharger sans toucher au code, exporter des variables d'environnement :

```bash
export DOQMENT_OLLAMA_HOST=http://localhost:11434
export DOQMENT_OLLAMA_TEXT_MODEL=mistral:7b-instruct
export DOQMENT_OLLAMA_VISION_MODEL=qwen2.5vl:7b
export DOQMENT_COLQWEN_DEVICE=cuda:0       # ou "cpu"
export DOQMENT_COLQWEN_DTYPE=bfloat16
```

---

## Structure du dépôt

```
TrAitementDoQment/
├── ingestion.py               ← canonique coéquipier 1 (intouché)
├── pipeline1.py               ← canonique coéquipier 1 (intouché)
├── pipeline_rag2.py           ← canonique coéquipier 2 (intouché)
├── pdf_ocr.py                 ← canonique coéquipier 2 (intouché)
├── Comparaison_OCR.py         ← canonique coéquipier 2 (intouché)
│
├── app.py                     ← interface Streamlit (3 vues)
├── conftest.py
├── requirements.txt
│
├── doqment/                   ← TOUT notre code, un seul package plat
│   ├── settings.py            ← config (Settings dataclass)
│   ├── llm.py                 ← client Ollama (texte + vision)
│   ├── ocr.py                 ← wrapper Tesseract
│   ├── phase1.py              ← Pipeline 1 (ingest_directory, ask_document, ask_database)
│   ├── phase2.py              ← Pipeline 2 (idem, multimodal)
│   ├── phase2_store.py        ← ColQwen2 + Qdrant + SQLite + rasterize
│   └── ui/                    ← 3 vues Streamlit (doc, db, ingest)
│
├── scripts/
│   ├── phase1.py              ← CLI Pipeline 1
│   └── phase2.py              ← CLI Pipeline 2
│
├── tests/                     ← 63 tests utiles, exécutés en <3 secondes
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

**`tesseract is not installed or it's not in your PATH`** — Installer le binaire Tesseract (voir Démarrage rapide). Sans Tesseract, Pipeline 1 mode `doc` marche uniquement si vous fournissez l'annotation ICDAR jumelle.

**`Connection refused` quand le pipeline appelle Ollama** — Démarrer le démon : `ollama serve` (souvent automatique après l'installation). Vérifier les modèles tirés : `ollama list`.

**`[transformers] Accessing __path__ from ...`** — Pré-avertissement de dépréciation de la version récente de `transformers`. Aucun effet sur le code.

**HuggingFace : `404 Not Found` sur `adapter_config.json` / `preprocessor_config.json`** — Normal. `sentence-transformers` cherche des fichiers optionnels qui n'existent pas pour MPNet.

**`Could not load library with AVX2 support` (FAISS)** — Le fallback générique s'active. Fonctionnel, performance légèrement plus basse.

**`AttributeError: 'OCREngine' object has no attribute '_model'`** — Bug du fichier canonique `ingestion.py:82-99`. Le mode `doc` Pipeline 1 contourne automatiquement en passant par `doqment/ocr.py` (Tesseract). Si vous voyez l'erreur, c'est que vous tournez une version antérieure — `pytest` doit afficher 62 passed.

**`ResponseError: model requires more system memory (12.5 GiB) than is available (...)`** — Qwen2.5-VL 7B requiert ~12,5 Go de RAM en CPU-only (~5 Go en GPU CUDA). Libérer de la RAM (fermer le navigateur, IDE, etc.) ou exécuter sur une machine équipée. Le projet est figé sur ce modèle ; ne pas le remplacer.

**`Found no NVIDIA driver on your system`** sur Pipeline 2 — ColQwen2 cherche un GPU CUDA. Sur machine CPU-only :
```bash
export DOQMENT_COLQWEN_DEVICE=cpu
export DOQMENT_COLQWEN_DTYPE=float32
```
L'encodage est ~50× plus lent qu'avec GPU (compter ~30 s par page) — réaliste sur 1-10 documents seulement.

---

## Trois fichiers canoniques cohabitent

Les coéquipiers ont livré du code Python qu'on conserve **byte-identique aux originaux** :

| Fichier | Apport | Qui s'en sert |
|---|---|---|
| `ingestion.py` + `pipeline1.py` | Indexation canonique SROIE (coéquipier 1) | `doqment/phase1.py` |
| `pipeline_rag2.py` | RAG complet avec REPL (coéquipier 2) | usage standalone |
| `pdf_ocr.py` | OCR Tesseract autonome (coéquipier 2) | extraction texte hors RAG |
| `Comparaison_OCR.py` | Benchmark Tesseract vs PaddleOCR (coéquipier 2) | diagnostic d'OCR |

---

## Limites assumées

- **L'index FAISS HNSW** (Pipeline 1) ne supporte pas la suppression individuelle — il faut réindexer pour retirer un document.
- **`ingestion.py:32`** : `BoundingBox.ymax` contient un bug (`max(self.x1, self.y2, self.y3, self.y4)` au lieu de `self.y1`). Contourné dans nos tests.
- **`ingestion.py:82-99`** : `OCREngine._load()` référence `self._model` qui n'existe pas. Contourné par `doqment/ocr.py` (Tesseract).
- **Pipeline 2 demande de la RAM** — Qwen2.5-VL 7B en CPU-only consomme ~12,5 Go (≈ 5 Go avec GPU CUDA). ColQwen2 ajoute ~6 Go. Sur machine ≤ 16 Go libre, fermer les autres applications avant de lancer `db`. Ce dimensionnement est assumé.
- **La Loi 25** — l'exécution locale ne couvre que l'aspect *technique*. Le consentement, le registre et le RPRP restent à instrumenter au niveau organisationnel.
