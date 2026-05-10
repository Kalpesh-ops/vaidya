# src/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from src.retrieval.retriever import VaidyaRetriever
from src.generation.generator import VaidyaGenerator

app = FastAPI(
    title="Vaidya API",
    description="Citation-grounded Ayurvedic knowledge from Caraka-Samhita",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

retriever = VaidyaRetriever()
generator = VaidyaGenerator()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    sthana_filter: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    response: str
    citations: list
    disclaimer: str
    confident: bool
    sources_used: int = 0


@app.get("/health")
def health():
    return {"status": "ok", "service": "Vaidya"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        retrieved = retriever.retrieve(
            query=req.query,
            top_k=req.top_k,
            sthana_filter=req.sthana_filter
        )
        confident = retriever.is_confident(retrieved)
        result = generator.generate(req.query, retrieved, confident)

        return QueryResponse(
            query=req.query,
            response=result["response"],
            citations=result["citations"],
            disclaimer=result["disclaimer"],
            confident=result["confident"],
            sources_used=result.get("sources_used", 0)
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)