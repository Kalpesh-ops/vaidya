# src/retrieval/retriever.py

from typing import List, Dict, Any, Optional
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "vaidya_caraka"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
CONFIDENCE_THRESHOLD = 0.60


class VaidyaRetriever:
    def __init__(self, qdrant_path: str = "data/index/qdrant_store"):
        logger.info("Loading retriever...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = QdrantClient(path=qdrant_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        sthana_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True
        )[0]

        search_filter = None
        if sthana_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            search_filter = Filter(
                must=[FieldCondition(
                    key="sthana",
                    match=MatchValue(value=sthana_filter)
                )]
            )

        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
            score_threshold=CONFIDENCE_THRESHOLD
        ).points

        results = []
        for hit in hits:
            results.append({
                "score": round(hit.score, 4),
                "chunk_id": hit.payload["chunk_id"],
                "sthana": hit.payload["sthana"],
                "adhyaya": hit.payload["adhyaya"],
                "text_sanskrit": hit.payload["text_sanskrit"],
                "text_english": hit.payload["text_english"],
                "text_hindi": hit.payload.get("text_hindi", ""),
                "citations": hit.payload["citations"],
                "primary_citation": hit.payload["primary_citation"],
            })

        if not results:
            logger.warning(f"No results above threshold {CONFIDENCE_THRESHOLD} for: {query}")

        return results

    def is_confident(self, results: List[Dict]) -> bool:
        return len(results) > 0 and results[0]["score"] >= CONFIDENCE_THRESHOLD