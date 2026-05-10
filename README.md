---
title: Vaidya
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# 🌿 Vaidya
**Citation-Grounded Ayurvedic RAG System for the Caraka-Saṃhitā**

[![Hackathon](https://img.shields.io/badge/Lablab.ai-AMD_Developer_Hackathon-blue)](https://lablab.ai)
[![Hardware](https://img.shields.io/badge/Compute-AMD_Instinct_MI300X-red)](#)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--72B--Instruct-purple)](#)

The internet is flooded with decontextualized, fabricated, and often dangerous "Ayurvedic" advice. Meanwhile, the actual classical texts—like the 8,400+ metrical verses of the *Caraka-Samhita*—remain locked behind language barriers and dense formatting. 

**Vaidya** is a highly disciplined Retrieval-Augmented Generation (RAG) system that democratizes this ancient knowledge without compromising integrity. It retrieves exact verses from the original Sanskrit text, translates them seamlessly into the user's language, and provides verifiable citations.

> **⚠️ Important Note Regarding the Live Demo**
> 
> Vaidya was fully developed, deployed, and tested on an **AMD Instinct™ MI300X GPU (192GB VRAM)** via a DigitalOcean droplet. This massive compute allowed us to serve the 72-Billion parameter Qwen2.5-72B-Instruct model with blazing-fast inference (~1,790 tokens/sec throughput). 
> 
> As per hackathon guidelines, because AMD GPU droplets bill continuously, the backend cloud instance hosting our LLM and vector database has been destroyed to conserve limited credits. As a result, this Hugging Face Space currently serves as a frontend landing page, and live search queries will timeout. 
> 
> **🎬 Please watch the Demo Video provided in our lablab.ai submission to see the complete, cross-lingual RAG system working flawlessly in real-time.**

## ✨ Key Features
* **Zero-Hallucination Policy:** Vaidya utilizes a strict Confidence Gate (cosine similarity threshold of 0.72). If a query lacks a direct textual source, the system refuses to answer rather than fabricating medical advice.
* **Traceable Citations:** Every factual claim is appended with its exact structural source (e.g., `[CS.Su.1.24]` -> *Caraka-Samhita, Sutrasthana, Chapter 1, Verse 24*).
* **Cross-Lingual Semantic Search:** Powered by `multilingual-e5-large`, users can query in Hindi, English, or Sanskrit, and the system dynamically maps the query to the original Sanskrit vectors.
* **Native Multilingual Synthesis:** Leveraging Qwen2.5, Vaidya detects the user's language and synthesizes the response in kind while preserving complex Ayurvedic terminology (Dosha, Dhatu, Agni).

## 🏗️ Architecture & Tech Stack

Vaidya is designed for maximum throughput and accuracy, powered natively by AMD hardware.

* **GPU Compute:** AMD Instinct™ MI300X (192GB HBM3) via DigitalOcean.
* **LLM Engine:** `Qwen2.5-72B-Instruct` served via `vLLM` (ROCm-optimized continuous batching).
* **Vector Database:** `Qdrant` for persistent, high-speed cosine similarity search.
* **Embeddings:** `intfloat/multilingual-e5-large` (1024 dimensions).
* **Backend:** FastAPI & LangChain.
* **Frontend:** Streamlit (Hosted on Hugging Face Spaces).

## 🚀 The Data Pipeline

We built a custom 5-stage pipeline to process the raw Devanagari OCR into a production-ready vector index:
1. **Parser (`parser.py`):** Extracts verses, maps metadata (Sthana, Adhyaya), and validates Devanagari integrity.
2. **Chunker (`chunker.py`):** Semantically groups verses while strictly preserving chapter boundaries.
3. **Translator (`translator.py`):** Pre-generates English and Hindi translations for all chunks using Qwen2.5-72B to eliminate runtime latency.
4. **Embedder (`embedder.py`):** Vectorizes the corpus into Qdrant.
5. **Generator (`generator.py`):** LangChain-powered retrieval and prompt formatting with mandatory disclaimer injection.

## 📊 Benchmarks & Performance
* **LLM Throughput:** ~1,790 tokens/sec generation throughput (vLLM on AMD MI300X).
* **Citation Completeness:** > 95% across all parsed chunks.
* **Cross-Lingual Retrieval:** Functional across English, Hindi, and Sanskrit queries.

## 💻 Local Setup (Development)

If you wish to run the pipeline locally (Requires AMD ROCm compatible GPU):

```bash
# 1. Clone the repository
git clone https://github.com/Kalpesh-ops/vaidya.git
cd vaidya

# 2. Setup Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.in

# 3. Start the ROCm vLLM Server (in a separate terminal)
docker run -it --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 rocm/vllm:latest \
  python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct --gpu-memory-utilization 0.90

# 4. Start the Backend API
python -m src.api.main

# 5. Launch the UI
streamlit run app.py

```

## ⚠️ Disclaimer

*Vaidya is an AI research system built for the Lablab.ai AMD Developer Hackathon. All output is sourced from classical texts for educational purposes only. Always consult a certified Ayurvedic practitioner or licensed medical professional before making any health decisions. This is not medical advice.*