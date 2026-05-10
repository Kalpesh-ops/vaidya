# run_all_benchmarks.py — Run after each stage completes

import json
import sys
from pathlib import Path
from loguru import logger

def check_stage1():
    p = Path("data/processed/parsed_verses.json")
    if not p.exists():
        return {"PASS": False, "error": "parsed_verses.json not found"}
    with open(p) as f:
        data = json.load(f)
    verses = data.get("verses", [])
    errors = data.get("errors", [])
    return {
        "stage": "Parser",
        "total_verses": len(verses),
        "total_errors": len(errors),
        "error_rate": round(len(errors) / max(len(verses), 1), 4),
        "PASS": len(verses) > 100 and len(errors) / max(len(verses), 1) < 0.10
    }

def check_stage2():
    p = Path("data/processed/chunks.jsonl")
    if not p.exists():
        return {"PASS": False, "error": "chunks.jsonl not found"}
    chunks = []
    with open(p) as f:
        for line in f:
            chunks.append(json.loads(line))
    avg_tokens = sum(c["token_count"] for c in chunks) / max(len(chunks), 1)
    missing_cite = sum(1 for c in chunks if not c["citations"])
    return {
        "stage": "Chunker",
        "total_chunks": len(chunks),
        "avg_tokens": round(avg_tokens, 2),
        "missing_citations": missing_cite,
        "PASS": len(chunks) > 0 and missing_cite == 0
    }

def check_stage3():
    p = Path("data/processed/translations_cache.jsonl")
    if not p.exists():
        return {"PASS": False, "error": "translations_cache.jsonl not found"}
    chunks = []
    with open(p) as f:
        for line in f:
            chunks.append(json.loads(line))
    empty_en = sum(1 for c in chunks if len(c.get("text_english", "")) < 10)
    empty_hi = sum(1 for c in chunks if len(c.get("text_hindi", "")) < 10)
    return {
        "stage": "Translator",
        "total_translated": len(chunks),
        "empty_english": empty_en,
        "empty_hindi": empty_hi,
        "PASS": len(chunks) > 0 and empty_en < len(chunks) * 0.05
    }

def check_stage4():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(path="data/index/qdrant_store")
        info = client.get_collection("vaidya_caraka")
        return {
            "stage": "Embedder",
            "vectors_indexed": info.points_count,
            "PASS": info.points_count > 0
        }
    except Exception as e:
        return {"PASS": False, "error": str(e)}

def check_api():
    try:
        import requests
        r = requests.get("http://localhost:8080/health", timeout=5)
        return {
            "stage": "API",
            "status": r.json(),
            "PASS": r.status_code == 200
        }
    except Exception as e:
        return {"PASS": False, "error": str(e)}

if __name__ == "__main__":
    stages = [check_stage1, check_stage2, check_stage3, check_stage4, check_api]
    all_pass = True

    print("\n" + "="*50)
    print("VAIDYA — FULL BENCHMARK REPORT")
    print("="*50)

    report_lines = ["# Vaidya Full Benchmark Report\n\n"]

    for fn in stages:
        result = fn()
        status = "✓ PASS" if result.get("PASS") else "✗ FAIL"
        all_pass = all_pass and result.get("PASS", False)

        print(f"\n[{status}] {result.get('stage', fn.__name__)}")
        for k, v in result.items():
            if k not in ("stage",):
                print(f"  {k}: {v}")

        report_lines.append(f"## {result.get('stage', fn.__name__)}: {status}\n\n")
        for k, v in result.items():
            report_lines.append(f"- **{k}**: {v}\n")
        report_lines.append("\n")

    print("\n" + "="*50)
    print(f"OVERALL: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    print("="*50)

    with open("benchmark_report.md", "w") as f:
        f.writelines(report_lines)
    print("\nSaved to benchmark_report.md")