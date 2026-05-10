# src/pipeline/embedder.py

import json
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType
)

COLLECTION_NAME = "vaidya_caraka"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_DIM = 1024
BATCH_SIZE = 32


class VaidyaEmbedder:
    def __init__(
        self,
        input_path: str,
        qdrant_path: str = "data/index/qdrant_store"
    ):
        self.input_path = Path(input_path)
        self.qdrant_path = qdrant_path

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info(f"Initializing Qdrant at: {qdrant_path}")
        self.client = QdrantClient(path=qdrant_path)
        self._setup_collection()

    def _setup_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME in existing:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
            return

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection: {COLLECTION_NAME}")

    def embed_and_index(self) -> int:
        logger.info(f"Loading translated chunks from: {self.input_path}")
        chunks = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))

        logger.info(f"Embedding {len(chunks)} chunks...")
        total_indexed = 0

        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
            batch = chunks[i:i + BATCH_SIZE]

            # Embed using multilingual query format for e5
            texts = [
                f"passage: {c['text_english']}" for c in batch
            ]
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                point = PointStruct(
                    id=total_indexed + j,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "sthana": chunk["sthana"],
                        "adhyaya": chunk["adhyaya"],
                        "text_sanskrit": chunk["text_sanskrit"],
                        "text_english": chunk["text_english"],
                        "text_hindi": chunk.get("text_hindi", ""),
                        "citations": chunk["citations"],
                        "verse_ids": chunk["verse_ids"],
                        "primary_citation": chunk["citations"][0] if chunk["citations"] else "",
                    }
                )
                points.append(point)

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            total_indexed += len(batch)

        logger.info(f"Indexed {total_indexed} chunks into Qdrant.")
        return total_indexed


class EmbedderBenchmark:
    def __init__(self, client: QdrantClient):
        self.client = client

    def run(self) -> Dict[str, Any]:
        results = {}
        info = self.client.get_collection(COLLECTION_NAME)
        results["total_vectors"] = info.points_count
        results["vector_dimension"] = EMBEDDING_DIM
        results["collection_name"] = COLLECTION_NAME

        # Test retrieval with a Sanskrit query
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)

        test_queries = [
            "query: what are the properties of vata dosha",
            "query: वात दोष के गुण क्या हैं",
            "query: pitta kapha treatment",
        ]

        retrieval_ok = True
        for q in test_queries:
            emb = model.encode([q], normalize_embeddings=True)
            hits = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=emb[0].tolist(),
                limit=3
            )
            if not hits:
                retrieval_ok = False
                logger.warning(f"No results for query: {q}")

        results["retrieval_functional"] = retrieval_ok
        results["PASS"] = (
            results["total_vectors"] > 0 and
            results["retrieval_functional"]
        )
        return results


if __name__ == "__main__":
    embedder = VaidyaEmbedder(
        input_path="data/processed/translations_cache.jsonl",
        qdrant_path="data/index/qdrant_store"
    )
    total = embedder.embed_and_index()

    bench = EmbedderBenchmark(embedder.client)
    results = bench.run()

    print("\n=== STAGE 4 BENCHMARK RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nOverall: {'✓ PASS' if results['PASS'] else '✗ FAIL'}")

    with open("benchmark_report.md", "a") as f:
        f.write("## Stage 4: Embedder\n\n")
        for k, v in results.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")