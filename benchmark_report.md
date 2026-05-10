# Vaidya Benchmark Report

## Stage 1: Parser

- **total_verses**: 8741
- **citation_completeness**: 1.0
- **sthana_distribution**: {'Sutrasthana': 1710, 'Vimanasthana': 341, 'Siddhisthana': 589, 'Nidanasthana': 277, 'Sharirasthana': 763, 'Cikitsasthana': 4756, 'Kalpasthana': 305}
- **avg_verse_length_chars**: 99.74
- **empty_or_short_verses**: 4
- **devanagari_ratio**: 0.8923
- **PASS**: True
## Stage 2: Chunker

- **total_chunks**: 1612
- **avg_chunk_tokens**: 222.07
- **citation_present_ratio**: 1.0
- **semantic_boundary_broken_ratio**: 0.0
- **outlier_chunks**: 0
- **PASS**: True

## Stage 3: Translator

- **total_translated_chunks**: 1612
- **empty_english_ratio**: 0.0
- **empty_hindi_ratio**: 0.0
- **keyword_presence_ratio**: 0.65
- **PASS**: True

## Stage 4: Embedder

- **total_vectors**: 1612
- **vector_dimension**: 1024
- **collection_name**: vaidya_caraka
- **retrieval_functional**: True
- **PASS**: True

