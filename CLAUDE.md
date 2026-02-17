# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) system implementing Dense + Sparse hybrid search based on [this Zenn article](https://zenn.dev/fp16/articles/aa48dcae23974e). Processes PDF documents with semantic + keyword-based search. Tested with both English (TMC4361A datasheet, 1148 chunks) and Japanese (日本国憲法, 27 chunks) documents.

## Key Commands

```bash
# Activate virtual environment (required before any Python commands)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Basic RAG system test with English document (TMC4361A datasheet)
python main.py

# Alpha parameter evaluation (0.0, 0.3, 0.5, 0.7, 1.0) — 5 queries
python evaluation.py

# Multi-model comparison — 6 small models × 4 alpha × 15 test cases (CPU, ~30min first run)
python model_comparison_fast.py

# Full multi-model comparison — 9 models including base variants (CPU, slower)
python model_comparison.py

# Japanese document tests
python test_ruri_kenpo.py        # ruri-v3-310m (requires sentencepiece)
python test_bgem3_kenpo.py       # BGE-M3 learned sparse (requires FlagEmbedding, ~2.2GB download, ~144s encode on CPU)

# Interactive search CUI — ruri-v3-310m × 日本国憲法 (dense/hybrid switchable)
python interactive_ruri.py

# Visualize alpha evaluation results (text-based charts, reads evaluation_results.json)

# Individual module verification
python pdf_processor.py   # PDF processing only
python rag_system.py      # Module load check
```

### GPU Evaluation (Google Colab)

`rag_gpu_evaluation.ipynb` — Upload to Google Colab with T4 GPU runtime. Evaluates 11 models (6 small + 5 large up to 1024-dim). Includes batch embedding, GPU memory tracking, and CPU vs GPU speed comparison. Upload `TMC4361A_datasheet_rev1.26_01.pdf` when prompted.

## Architecture

### Core Modules

- **`pdf_processor.py`** (`PDFProcessor`) — PDF text extraction via PyMuPDF, chunking with overlap
- **`rag_system.py`** (`HybridRAGSystem`) — Dense/Sparse embedding, ChromaDB storage, hybrid search
- **`test_cases.py`** (`TEST_CASES`) — 15 test case definitions (5 categories × 3 queries)

### Two Distinct Search Architectures

**1. Standard pipeline** (used by `main.py`, `evaluation.py`, `model_comparison*.py`, `test_ruri_kenpo.py`):
```
PDFProcessor → chunks → HybridRAGSystem.add_documents() → ChromaDB
Query → dense_search() + sparse_search() → min_max_normalize() → weighted fusion
```
- Dense: SentenceTransformer (configurable model)
- Sparse: Jaccard similarity on whitespace-tokenized sets (`text.lower().split()`)

**2. BGE-M3 standalone pipeline** (`test_bgem3_kenpo.py` only):
```
PDFProcessor → chunks → BGEM3FlagModel.encode(return_sparse=True)
Query → cosine similarity (dense) + compute_lexical_matching_score() (sparse) → weighted fusion
```
- Dense: BGE-M3 1024-dim vectors
- Sparse: Learned lexical weights (subword-tokenized, works for Japanese)
- Does NOT use `HybridRAGSystem` or ChromaDB — implements its own similarity computation

### Module Import Graph

```
main.py, evaluation.py, test_ruri_kenpo.py, interactive_ruri.py  →  pdf_processor + rag_system
model_comparison.py, model_comparison_fast.py  →  pdf_processor + rag_system + test_cases
test_bgem3_kenpo.py  →  pdf_processor + FlagEmbedding (standalone, no rag_system)
visualize_results.py  →  reads JSON files only (no module imports)
```

### Hybrid Search Algorithm (`rag_system.py`)

1. Retrieve top-K×2 candidates from Dense (ChromaDB cosine) and Sparse (Jaccard) independently
2. Merge candidate sets (union of doc IDs)
3. Min-Max normalize scores per modality independently
4. Combine: `alpha × dense_norm + (1-alpha) × sparse_norm`
5. Return top-K by combined score

### Test Case Categories (`test_cases.py`)

5 categories × 3 queries = 15 test cases for English TMC4361A evaluation:
- **factual_numeric** (easy): specific values — voltage, temperature, SPI frequency
- **conceptual** (medium): mechanism/functionality understanding
- **technical_terms** (easy-hard): register names, feature acronyms
- **multi_aspect** (hard): queries combining multiple topics
- **semantic_similarity** (medium-hard): indirect/paraphrased expressions

Some models require query prefixes — bge: `"Represent this sentence: "`, e5: `"query: "`.

## Critical Parameters

- **`alpha=0.7`**: Confirmed optimal for English across all 6 tested CPU models. Dense-focused hybrid outperforms both pure Dense and pure Sparse.
- **`alpha=1.0` for Japanese with standard pipeline**: Sparse search with whitespace tokenization hurts Japanese — use Dense-only or Dense-heavy.
- **`alpha=0.7` for Japanese with BGE-M3**: Learned sparse works for Japanese — hybrid achieves avg 0.98.
- **`alpha=0.0`**: Sparse-only keyword match scores are model-independent (same Jaccard for all models).
- ChromaDB collections are **reset on each run** — documents must be re-embedded every time.
- Chunk parameters: 512 chars, 50 overlap, extend to next newline within 100 chars.

## Key Implementation Notes

- **Sparse search does not work for Japanese text** in `rag_system.py` — the tokenizer uses `text.lower().split()` (whitespace split), producing very few tokens for Japanese. BGE-M3 learned sparse (`test_bgem3_kenpo.py`) solves this with subword-tokenized lexical weights.
- Sparse search uses Jaccard similarity (intersection/union of token sets), not BM25. This simplification causes table-of-contents chunks (high keyword density) to rank disproportionately high.
- **Top Score metric is misleading at alpha=0.0** — Min-Max normalization always sets the top sparse result to 1.0, making sparse-only appear to have perfect scores regardless of actual relevance.
- `HybridRAGSystem.__init__` accepts `model_name` for any SentenceTransformer model. The GPU notebook version also accepts a `device` parameter (`"cuda"` or `"cpu"`).
- SentenceTransformer models cache in `~/.cache/huggingface/hub/`.
- **ruri-v3-310m** (`cl-nagoya/ruri-v3-310m`, 768-dim) requires `sentencepiece` package. Dense scores on Japanese: 0.80–0.87.
- **Python 3.9 + torch constraint**: `transformers<4.51` is pinned because torch >= 2.6 is unavailable for Python 3.9 (CVE-2025-32434). Remove this constraint if upgrading to Python 3.10+.
- Evaluation metrics: **Keyword Match Score** = percentage of expected keywords found in Top-3 result texts; **Top Score** = combined score of the highest-ranked result.

## File Paths

All scripts use hardcoded absolute paths to `/Users/seigo/Desktop/working/RagTest/`. Update these if the repo moves:
- English PDF: `TMC4361A_datasheet_rev1.26_01.pdf`
- Japanese PDF: `nihonkokukenpou.pdf`
- Results: `results_summary.json`, `evaluation_results.json`, `model_comparison_results.json`, `ruri_kenpo_results.json`, `bgem3_kenpo_results.json`

## Differences from Zenn Article

| Component | Zenn Article | This Implementation |
|-----------|-------------|---------------------|
| Dense Model | ruri-v3-310m (768-dim) | all-MiniLM-L6-v2 (384-dim) default, ruri tested separately |
| Sparse Method | bge-m3 learned sparse | Jaccard (default), bge-m3 learned sparse (`test_bgem3_kenpo.py`) |
| Evaluation Metrics | P@K, MRR, NDCG | Keyword Match Score, Top Score |
| Hardware | GPU (RTX 4090) | CPU (macOS) + GPU (Colab T4) |
