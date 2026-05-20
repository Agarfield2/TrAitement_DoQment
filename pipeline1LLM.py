"""
Pipeline RAG — Retrieval + Mistral 7B

Usage :
    python pipeline1LLM.py --download            # necesssaire pour le premier run
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⡤⠤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠴⠟⠋⠀⠀⠀⠀⠈⠙⠳⣦⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠞⠁⠀⠀⠀⢀⣀⣀⣤⡴⢂⣀⠀⠻⡆⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠏⣀⣤⣴⡿⣋⣤⣍⠥⠾⠟⠛⠃⠀⠀⣷⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠈⠉⠉⠉⠀⠀⠀⠀⠐⠚⠛⠙⠀⠀⣿⡀⠀⠀⠀
⠀⠀⠀⣀⡴⠞⠋⠉⠀⠉⠙⠓⠦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡄⠠⠶⠟⣛⣓⣀⢀⣰⣿⣶⣶⠀⠀⠀⢻⡇⠀⠀⠀
⠀⣠⡾⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢶⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣧⠀⠰⣿⡿⡿⡇⢸⠉⠉⠉⠀⠀⠀⠀⢸⡇⠀⠀⠀
⠾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣶⣦⣀⡀⠀⠀⠀⠀⢻⡆⠀⠀⠀⢀⡇⠈⢷⡄⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀
⠀⠒⢲⣶⣶⣶⣶⣒⡲⠶⠦⠤⠤⣤⢤⠄⣀⠀⢳⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣷⠈⠉⠓⠒⠶⢤⣬⣿⠀⠀⠀⠸⣧⡤⠴⠇⠀⠀⠀⠀⣠⣿⡇⠀⠀⠀
⢀⡾⠟⣛⡉⠙⠓⠻⠟⠉⢉⣽⠿⠻⠯⢵⣦⠉⠀⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣦⣄⠀⠀⠀⠀⠀⠙⣶⡀⠀⣠⣰⣖⣒⡦⣄⡀⠀⣸⣿⣿⠍⠀⠠⣤
⠈⠀⠉⠉⠉⢷⣤⡀⠀⠀⠀⠀⠀⣠⠴⠒⢮⣄⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠳⣄⠀⠀⠀⠀⠈⢷⣼⣿⣿⣿⣿⣷⡌⢹⣶⣿⣿⠛⠀⠀⠀⠀
⠛⠛⣟⣿⣿⢿⠶⠦⣤⠄⢲⣤⠤⢬⡤⡤⠤⠤⣤⣤⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢆⠀⠀⠀⠀⠻⣇⠙⠛⡛⠁⠀⢈⣴⡿⠃⠀⠀⠀⠀⠀
⠀⠀⡻⠿⣿⣶⡄⠀⣼⡖⢾⣧⢠⣿⣷⣦⠆⠀⠀⣿⣿⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⣄⠀⠀⠀⠘⠿⣦⣄⣀⣠⣾⠋⠀⠀⠀⠀⠀⠀⠀
⣤⣀⣛⣳⣶⣿⣣⡾⡟⠀⠀⢿⣿⣿⣱⣷⣾⣂⣼⠿⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢳⡄⠀⠀⠀⠈⠋⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠉⠉⠉⠉⠉⠉⠀⢀⡇⠀⠀⠘⡇⠉⠉⠉⠉⠁⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢾⡁⠀⠀⢀⡇⠀⠀⠀⠀⠀⠀⢠⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢠⡍⠙⠋⠁⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⡶⠛⣹⣯⠰⣿⣆⠀⠀⠀⠀⢀⣼⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣧⠀⠀⠀⠀⣿⣇⣽⣿⣿⣦⣭⡟⣧⠀⠀⢰⡦⡿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⠆⠀⠀⢀⣿⣿⣿⣿⣿⣿⡏⡟⠓⡄⣶⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣥⠈⣶⡆⣼⣿⣾⡎⠿⠿⠟⠁⡇⢸⡿⢿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢿⣿⣿⣷⡇⣾⢿⡗⠶⠴⠆⠈⢠⠞⣳⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠂⠙⣿⣯⣷⣿⠸⣿⠀⠀⣤⠀⣈⣴⠟⠂⣻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠈⡿⣿⣀⡙⡅⠀⢰⣧⣿⡏⠀⢠⢿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    
    python pipeline1LLM.py                       # demo sur 3 fichiers du test set
    python pipeline1LLM.py --n-demo n            # demo sur n fichiers
    python pipeline1LLM.py --eval                # evaluation complete test set
    python pipeline1LLM.py --eval --max-eval n  # evaluation sur n docs
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Structures de donnees

@dataclass
class BoundingBox:
    x1: float; y1: float
    x2: float; y2: float
    x3: float; y3: float
    x4: float; y4: float
    
@dataclass
class Passage:
    passage_id: str
    text: str
    source_file: str
    page_number: int
    bboxes: list[BoundingBox]
    avg_confidence: float
    entities: Optional[dict] = None

# Train / test split

def train_test_split(
    metadata: list[dict],
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    all_files = sorted({m["source_file"] for m in metadata})
    rng = random.Random(random_seed)
    rng.shuffle(all_files)

    cut = int(len(all_files) * (1 - test_ratio))
    train_files = set(all_files[:cut])
    test_files  = set(all_files[cut:])

    train = [m for m in metadata if m["source_file"] in train_files]
    test  = [m for m in metadata if m["source_file"] in test_files]

    print(
        f"Split : {len(train_files)} docs ({len(train)} passages) train  |  "
        f"{len(test_files)} docs ({len(test)} passages) test"
    )
    return train, test

# Retrieval (FAISS)

class RetrievalModule:
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    DIM = 768

    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            print("Chargement du modele d'embedding...")
            self.model = SentenceTransformer(self.MODEL_NAME)
        self.index = faiss.IndexFlatIP(self.DIM)
        self.metadata: list[dict] = []

    def build(self, raw_passages: list[dict]):
        passages = []
        for item in raw_passages:
            bboxes = [BoundingBox(**b) for b in item.get("bboxes", [])]
            passages.append(Passage(
                passage_id=item["passage_id"],
                text=item["text"],
                source_file=item["source_file"],
                page_number=item["page_number"],
                bboxes=bboxes,
                avg_confidence=item["avg_confidence"],
                entities=item.get("entities"),
            ))

        embeddings = self.model.encode(
            [p.text for p in passages],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        self.index.add(embeddings)
        self.metadata = [
            {
                "passage_id": p.passage_id,
                "text": p.text,
                "source_file": p.source_file,
                "page_number": p.page_number,
                "avg_confidence": p.avg_confidence,
                "entities": p.entities,
                "bboxes": [asdict(b) for b in p.bboxes],
            }
            for p in passages
        ]

    def search(self, query: str, k: int = 3) -> list[dict]:
        k = min(k, self.index.ntotal)
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)
        return sorted(results, key=lambda r: r["score"], reverse=True)


# Modele Mistral 7B

MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / MODEL_FILE


def download_model():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")
    if MODEL_PATH.exists():
        print(f"Modele deja present : {MODEL_PATH}")
        return
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"Telechargement de {MODEL_FILE} (~4.1 GB)...")
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    print("Telechargement termine.")


class MistralLLM:
    def __init__(self, n_ctx: int = 4096, n_threads: int = 4,
                 n_gpu_layers: int = 0, verbose: bool = False):
        self._llm = None
        self._init_kwargs = dict(
            model_path=str(MODEL_PATH),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def _load(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("pip install llama-cpp-python")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modele introuvable : {MODEL_PATH}\n"
                "Lancer d'abord : python pipeline1LLM.py --download"
            )
        print("Chargement de Mistral 7B...")
        self._llm = Llama(**self._init_kwargs)

    def _build_prompt(self, question: str, passages: list[dict]) -> str:
        context = "\n\n".join(
            f"[Passage {i}]\n{p['text']}"
            for i, p in enumerate(passages, 1)
        )
        return (
            "[INST] "
            "You extract a single field from a scanned receipt. "
            "Reply with the extracted value ONLY — no words, no punctuation, no explanation.\n"
            "Rules:\n"
            "- TOTAL AMOUNT = the final amount due (TOTAL, TOTAL SALES, TOTAL AMOUNT PAYABLE). "
            "Ignore CASH RECEIVED and CHANGE. "
            "If OCR shows spaces inside a number (e.g. '1 19.34'), remove the space (119.34).\n"
            "- TAX AMOUNT = the GST/tax value in the TAX column, not the pre-tax subtotal.\n"
            "- DATE = transaction date only (DD/MM/YYYY or similar). Ignore the time.\n"
            "- COMPANY NAME = business name only, no registration number or address.\n"
            "- If the field is absent, reply: Not found\n\n"
            f"RECEIPT PASSAGES:\n{context}\n\n"
            f"FIELD TO EXTRACT: {question} "
            "[/INST]"
        )

    def generate(self, question: str, passages: list[dict]) -> str:
        self._load()
        output = self._llm(
            self._build_prompt(question, passages),
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=["\n", "(", "Assuming", "Note", "</s>"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()


# Pipeline principal

DEMO_QUERIES: list[tuple[str, str]] = [
    ("What is the total amount?", "total"),
    ("What is the company name?", "company"),
    ("What is the receipt date?", "date"),
    ("What is the tax amount?",   "tax"),
]


class EmbeddingPipeline:
    def __init__(
        self,
        metadata_path: str = "data/processed/metadata.pkl",
        split: str = "test",
        top_k: int = 3,
        n_threads: int = 4,
        random_seed: int = 42,
        test_ratio: float = 0.2,
    ):
        self.top_k = top_k
        self.random_seed = random_seed

        with open(metadata_path, "rb") as f:
            all_metadata = pickle.load(f)

        train_meta, test_meta = train_test_split(
            all_metadata, test_ratio=test_ratio, random_seed=random_seed
        )

        self.split_meta = train_meta if split == "train" else test_meta
        self.split_name = split

        self._test_by_file: dict[str, list[dict]] = {}
        for m in test_meta:
            self._test_by_file.setdefault(m["source_file"], []).append(m)

        self.retrieval = RetrievalModule()
        self.llm = MistralLLM(n_threads=n_threads)

    def _local_retrieval(self, passages: list[dict]) -> RetrievalModule:
        local = RetrievalModule(model=self.retrieval.model)
        local.build(passages)
        return local

    def run_demo(self, n_files: int = 3):
        file_map: dict[str, list[dict]] = {}
        for m in self.split_meta:
            file_map.setdefault(m["source_file"], []).append(m)

        chosen = random.Random(self.random_seed).sample(
            sorted(file_map), min(n_files, len(file_map))
        )

        for src in chosen:
            file_passages = file_map[src]
            local = self._local_retrieval(file_passages)
            gt = file_passages[0].get("entities") or {}

            print(f"\n{'-' * 60}")
            print(f"Fichier : {Path(src).name}  ({len(file_passages)} passages, {self.split_name} set)")
            if gt:
                print(f"Ground truth : {json.dumps(gt, ensure_ascii=False)}")
            print()

            for question, field in DEMO_QUERIES:
                passages = local.search(question, k=self.top_k)
                answer   = self.llm.generate(question, passages)
                gt_val   = gt.get(field, "—")
                print(f"  {question}")
                print(f"    LLM      : {answer}")
                print(f"    Expected : {gt_val}")

    def evaluate(self, max_docs: Optional[int] = None):
        test_files = sorted(self._test_by_file.keys())
        if max_docs:
            test_files = test_files[:max_docs]

        print(f"\nEvaluation sur {len(test_files)} fichiers du test set...")

        counts  = {field: 0 for _, field in DEMO_QUERIES}
        correct = {field: 0 for _, field in DEMO_QUERIES}

        for src in test_files:
            file_passages = self._test_by_file[src]
            gt = file_passages[0].get("entities") or {}
            local = self._local_retrieval(file_passages)

            for question, field in DEMO_QUERIES:
                gt_val = gt.get(field)
                if gt_val is None:
                    continue
                answer = self.llm.generate(question, local.search(question, k=self.top_k))
                counts[field] += 1
                if answer.strip().lower() == str(gt_val).strip().lower():
                    correct[field] += 1

        print()
        total_c = total_n = 0
        for _, field in DEMO_QUERIES:
            n, c = counts[field], correct[field]
            print(f"  {field:<12} {c}/{n}  ({c/n*100:.1f}%)" if n else f"  {field:<12} no data")
            total_c += c
            total_n += n
        print(f"  {'---'}")
        print(f"  {'total':<12} {total_c}/{total_n}  ({total_c/total_n*100:.1f}%)" if total_n else "  no data")

# Point d'entrée

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline RAG — Mistral 7B + FAISS")
    parser.add_argument("--download",   action="store_true")
    parser.add_argument("--metadata",   default="data/processed/metadata.pkl")
    parser.add_argument("--split",      default="test", choices=["train", "test"])
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--top-k",      type=int, default=3)
    parser.add_argument("--threads",    type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n-demo",     type=int, default=3)
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--max-eval",   type=int, default=None)
    args = parser.parse_args()

    if args.download:
        download_model()
    else:
        pipeline = EmbeddingPipeline(
            metadata_path=args.metadata,
            split=args.split,
            top_k=args.top_k,
            n_threads=args.threads,
            random_seed=args.seed,
            test_ratio=args.test_ratio,
        )
        if args.eval:
            pipeline.evaluate(max_docs=args.max_eval)
        else:
            pipeline.run_demo(n_files=args.n_demo)