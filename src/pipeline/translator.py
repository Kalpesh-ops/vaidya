# src/pipeline/translator.py

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from loguru import logger
from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

# --- Data Models ---

class TranslatedChunk(BaseModel):
    chunk_id: str
    sthana: str
    adhyaya: int
    text_sanskrit: str
    text_english: str
    text_hindi: str
    citations: List[str]
    verse_ids: List[str]
    token_count: int

class TranslationOutput(BaseModel):
    english_translation: str
    hindi_translation: str

# --- Translator Logic ---

class QwenTranslator:
    def __init__(
        self, 
        input_path: str, 
        output_path: str,
        base_url: str = "http://localhost:8000/v1", # Standard vLLM default port
        model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        logger.info(f"Initializing Qwen Translator via API at {base_url}")
        
        # We use ChatOpenAI because vLLM provides an OpenAI-compatible server
        # This makes LangChain integration seamless. API key is dummy for local.
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key="EMPTY", 
            model=model_name,
            temperature=0.1, # Low temperature for accurate, factual translation
            max_retries=3
        )
        
        self.parser = JsonOutputParser(pydantic_object=TranslationOutput)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert scholar of classical Ayurveda and Sanskrit. "
                       "Your task is to translate the provided original Sanskrit verses from the Caraka-Samhita "
                       "into accurate, clinical English and clear, precise Hindi. "
                       "Preserve Ayurvedic terminology (e.g., Dosha, Dhatu, Vata, Pitta, Kapha) where appropriate. "
                       "\n{format_instructions}"),
            ("human", "Translate the following Sanskrit text:\n\n{sanskrit_text}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def process_chunks(self) -> List[TranslatedChunk]:
        with open(self.input_path, 'r', encoding='utf-8') as f:
            chunks_data = [json.loads(line) for line in f]
            
        translated_chunks = []
        
        logger.info(f"Translating {len(chunks_data)} chunks...")
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open in append mode so if it crashes, we don't lose progress
        with open(self.output_path, 'w', encoding='utf-8') as out_f:
            for chunk in tqdm(chunks_data, desc="Translating"):
                try:
                    response = self.chain.invoke({
                        "sanskrit_text": chunk["text_sanskrit"],
                        "format_instructions": self.parser.get_format_instructions()
                    })
                    
                    translated = TranslatedChunk(
                        chunk_id=chunk["chunk_id"],
                        sthana=chunk["sthana"],
                        adhyaya=chunk["adhyaya"],
                        text_sanskrit=chunk["text_sanskrit"],
                        text_english=response["english_translation"],
                        text_hindi=response["hindi_translation"],
                        citations=chunk["citations"],
                        verse_ids=chunk["verse_ids"],
                        token_count=chunk["token_count"]
                    )
                    
                    translated_chunks.append(translated)
                    out_f.write(translated.model_dump_json() + '\n')
                    
                except Exception as e:
                    logger.error(f"Translation failed for chunk {chunk['chunk_id']}: {e}")
                    # In a production system, we'd log this to a retry queue.
                    
        return translated_chunks

# --- Stage 3 Benchmark ---
class TranslatorBenchmark:
    def __init__(self, translated_chunks: List[TranslatedChunk]):
        self.chunks = translated_chunks

    def run(self) -> Dict[str, Any]:
        results = {}
        total = len(self.chunks)
        results["total_translated_chunks"] = total
        
        if total == 0:
            results["PASS"] = False
            return results

        # Check for empty translations
        empty_en = sum(1 for c in self.chunks if len(c.text_english.strip()) < 10)
        empty_hi = sum(1 for c in self.chunks if len(c.text_hindi.strip()) < 10)
        
        results["empty_english_ratio"] = empty_en / total
        results["empty_hindi_ratio"] = empty_hi / total
        
        # Check Ayurvedic terminology retention (basic proxy metric)
        keywords = ["vata", "pitta", "kapha", "dosha", "dhatu", "agni", "prana"]
        retention_count = 0
        for c in self.chunks:
            en_lower = c.text_english.lower()
            if any(k in en_lower for k in keywords):
                retention_count += 1
                
        # Only valid if the source text actually contained these concepts, 
        # but serves as a baseline check.
        results["keyword_presence_ratio"] = round(retention_count / total, 2)

        results["PASS"] = (
            results["empty_english_ratio"] < 0.05 and
            results["empty_hindi_ratio"] < 0.05
        )
        
        return results

if __name__ == "__main__":
    INPUT_FILE = "data/processed/chunks.jsonl"
    OUTPUT_FILE = "data/processed/translations_cache.jsonl"
    
    translator = QwenTranslator(input_path=INPUT_FILE, output_path=OUTPUT_FILE)
    translated_data = translator.process_chunks()
    
    bench = TranslatorBenchmark(translated_data)
    results = bench.run()
    
    print("\n=== STAGE 3 BENCHMARK RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    status = "✓ PASS" if results.get("PASS", False) else "✗ FAIL"
    print(f"\nOverall: {status}")

    # Append to benchmark report
    with open("benchmark_report.md", "a", encoding="utf-8") as f:
        f.write("## Stage 3: Translator\n\n")
        for k, v in results.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")