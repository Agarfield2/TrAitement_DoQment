from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float


@dataclass
class Passage:
    passage_id: str
    text: str
    source_file: str
    page_number: int
    bboxes: list[BoundingBox]
    avg_confidence: float
    entities: Optional[dict] = None


class EmbeddingPipeline:

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    DIM = 768

    def __init__(
        self,
        metadata_path: str = "data/processed/metadata.pkl",
        index_dir: str = "data/faiss_index",
        max_docs: int = 200,
    ):

        self.metadata_path = Path(metadata_path)
        self.index_dir = Path(index_dir)
        self.max_docs = max_docs

        print("Chargement du modèle d'embedding...")
        self.model = SentenceTransformer(self.MODEL_NAME)

        print("Initialisation de FAISS...")
        self.index = faiss.IndexHNSWFlat(self.DIM, 32)
        self.index.hnsw.efSearch = 64

        self.passages: list[Passage] = []
        self.metadata: list[dict] = []

    def load_preprocessed_data(self):

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {self.metadata_path}"
            )

        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)

        metadata = metadata[: self.max_docs]

        passages = []

        for item in metadata:

            bboxes = [
                BoundingBox(**bbox)
                for bbox in item.get("bboxes", [])
            ]

            passages.append(
                Passage(
                    passage_id=item["passage_id"],
                    text=item["text"],
                    source_file=item["source_file"],
                    page_number=item["page_number"],
                    bboxes=bboxes,
                    avg_confidence=item["avg_confidence"],
                    entities=item.get("entities"),
                )
            )

        self.passages = passages

        print(f"{len(self.passages)} passages chargés")

    def encode_passages(self):

        texts = [p.text for p in self.passages]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.astype("float32")

    def build_faiss_index(self, embeddings: np.ndarray):

        self.index.add(embeddings)

        for p in self.passages:

            self.metadata.append({
                "passage_id": p.passage_id,
                "text": p.text,
                "source_file": p.source_file,
                "page_number": p.page_number,
                "avg_confidence": p.avg_confidence,
                "entities": p.entities,
                "bboxes": [asdict(b) for b in p.bboxes],
            })

        print(f"Index FAISS construit : {self.index.ntotal} passages")

    def save(self):

        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(self.index_dir / "index.faiss")
        )

        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"Index sauvegardé dans : {self.index_dir}")

    def search(
        self,
        query: str,
        k: int = 5,
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < 0:
                continue

            result = dict(self.metadata[idx])
            result["score"] = float(score)

            results.append(result)

        return results

    def run_demo_queries(self):

        queries = [
            "What is the total amount?",
            "What is the company name?",
            "What is the receipt date?",
            "tax amount",
        ]

        for query in queries:

            print("\n" + "=" * 80)
            print(f"QUESTION : {query}")
            print("=" * 80)

            results = self.search(query, k=3)

            for i, r in enumerate(results, start=1):

                print(f"\nRésultat #{i}")
                print(f"Score      : {r['score']:.4f}")
                print(f"Document   : {Path(r['source_file']).name}")
                print(f"Page       : {r['page_number']}")

                if r["entities"]:
                    print("\nEntities :")
                    print(json.dumps(r["entities"], indent=2))

                print("\nPassage :")
                print(r["text"][:500])

    def run(self):

        print("\nChargement des passages préprocessés...")
        self.load_preprocessed_data()

        print("\nCalcul des embeddings...")
        embeddings = self.encode_passages()

        print("\nConstruction de l'index FAISS...")
        self.build_faiss_index(embeddings)

        print("\nSauvegarde...")
        self.save()

        print("\nRecherche de test...")
        self.run_demo_queries()


if __name__ == "__main__":

    pipeline = EmbeddingPipeline(
        metadata_path="data/processed/metadata.pkl",
        index_dir="data/faiss_index",
        max_docs=200,
    )

    pipeline.run()