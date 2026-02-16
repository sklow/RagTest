# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG (Retrieval-Augmented Generation) system implementing Dense + Sparse hybrid search based on [this Zenn article](https://zenn.dev/fp16/articles/aa48dcae23974e). Processes PDF documents with semantic + keyword-based search. Tested with both English (TMC4361A datasheet) and Japanese (日本国憲法) documents.

## Key Commands

```bash
# Activate virtual environment (required before any Python commands)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Basic RAG system test with English document (TMC4361A datasheet)
python main.py

# Japanese document test with ruri-v3-310m (requires sentencepiece)
python test_ruri_kenpo.py

# Japanese document test with BGE-M3 learned sparse (requires FlagEmbedding)
# Uses BAAI/bge-m3 for both Dense (1024-dim) and Learned Sparse (lexical weights)
# WARNING: First run downloads ~2.2GB model, encoding is slow on CPU (~144s for 27 chunks)
python test_bgem3_kenpo.py

# Alpha parameter evaluation (0.0, 0.3, 0.5, 0.7, 1.0)
python evaluation.py

# Multi-model comparison (6 SentenceTransformer models × 4 alpha values × 15 test cases)
# WARNING: Downloads models from HuggingFace on first run (~30min total on CPU)
python model_comparison_fast.py

# Visualize alpha evaluation results (text-based charts)
python visualize_results.py

# Individual module verification
python pdf_processor.py   # PDF processing only
python rag_system.py      # Module load check
```

### GPU Evaluation (Google Colab)

`rag_gpu_evaluation.ipynb` — Upload to Google Colab with T4 GPU runtime. Evaluates 11 models (6 small + 5 large up to 1024-dim). Includes batch embedding, GPU memory tracking, and CPU vs GPU speed comparison. Upload `TMC4361A_datasheet_rev1.26_01.pdf` when prompted.

## Architecture

### Three-Stage Pipeline

1. **PDF Processing** (`pdf_processor.py`) — Extracts text via PyMuPDF, splits into overlapping chunks (512 chars, 50 overlap). Extends chunk boundaries to next newline within 100 chars to avoid mid-sentence cuts.

2. **Embedding & Storage** (`rag_system.py` → `HybridRAGSystem`) — Dense embedding via SentenceTransformer (default: `all-MiniLM-L6-v2`, 384-dim). Sparse embedding via TF (term frequency normalized by token count), similarity via Jaccard (intersection/union of token sets). Storage in ChromaDB with cosine similarity. **Alternative**: `test_bgem3_kenpo.py` uses BGE-M3 (`BAAI/bge-m3`) via FlagEmbedding for both Dense (1024-dim) and learned Sparse (lexical weights) — does not use `HybridRAGSystem` or ChromaDB.

3. **Hybrid Search** (`rag_system.py` → `hybrid_search()`) — Retrieves top-2K from both Dense and Sparse, applies Min-Max normalization independently, then combines: `alpha * dense_norm + (1-alpha) * sparse_norm`.

### Model Comparison Framework

`model_comparison_fast.py` (CPU) and `rag_gpu_evaluation.ipynb` (GPU) import test cases from `test_cases.py` — 5 categories × 3 queries each:
- **factual_numeric** (easy): specific values like voltage, temperature
- **conceptual** (medium): mechanism/functionality understanding
- **technical_terms** (easy-hard): register names, feature acronyms
- **multi_aspect** (hard): queries combining multiple topics
- **semantic_similarity** (medium-hard): indirect/paraphrased expressions

Some models require query prefixes — bge: `"Represent this sentence: "`, e5: `"query: "`. Results go to `model_comparison_results.json` (CPU) or `gpu_model_comparison_results.json` (GPU).

### Data Flow

`PDFProcessor.process_pdf()` → chunks list → `HybridRAGSystem.add_documents()` → ChromaDB. Queries go through `dense_search()` and `sparse_search()` independently, then `hybrid_search()` merges via weighted fusion.

## Critical Parameters

- **`alpha=0.7`**: Confirmed optimal across all 6 tested models (CPU). Dense-focused hybrid outperforms both pure Dense and pure Sparse.
- **`alpha=0.0`**: Sparse-only scores are model-independent (same Jaccard for all models).
- ChromaDB collections are **reset on each run** — documents must be re-embedded every time.

## File Paths

All scripts use hardcoded absolute paths to `/Users/seigo/Desktop/working/RagTest/`. Update these if the repo moves:
- English PDF: `TMC4361A_datasheet_rev1.26_01.pdf`
- Japanese PDF: `nihonkokukenpou.pdf`
- Results: `results_summary.json`, `evaluation_results.json`, `model_comparison_results.json`, `ruri_kenpo_results.json`, `bgem3_kenpo_results.json`

## Key Implementation Notes

- Sparse search in `rag_system.py` uses Jaccard similarity (intersection/union of token sets), not true BM25. This is a known simplification — BM25 would significantly improve Sparse quality.
- **Sparse search in `rag_system.py` does not work for Japanese text** — the tokenizer uses `text.lower().split()` (whitespace split), which produces very few tokens for Japanese. Dense-only (`alpha=1.0`) or Dense-heavy hybrid is required for Japanese documents with this system.
- **BGE-M3 learned sparse solves the Japanese Sparse problem** — `test_bgem3_kenpo.py` uses `BGEM3FlagModel` from FlagEmbedding, which provides subword-tokenized lexical weights that work for Japanese. Results: Dense avg 0.61, Sparse avg 0.14, Hybrid(0.7) avg 0.98. Uses `model.encode(return_sparse=True)` for sparse vectors and `model.compute_lexical_matching_score()` for similarity.
- `HybridRAGSystem.__init__` accepts `model_name` to swap SentenceTransformer models. Models with HuggingFace org prefixes (e.g., `BAAI/bge-small-en-v1.5`) work directly. The GPU notebook version also accepts a `device` parameter (`"cuda"` or `"cpu"`).
- SentenceTransformer models cache in `~/.cache/huggingface/hub/`. First download can be slow with flaky connections (retry logic is built into HuggingFace).
- **ruri-v3-310m** (`cl-nagoya/ruri-v3-310m`, 768-dim) requires `sentencepiece` package. Dense scores on Japanese text are 0.80–0.87 across diverse query types.
- **Python 3.9 + torch constraint**: torch >= 2.6 is not available for Python 3.9. transformers >= 4.51 enforces a torch >= 2.6 requirement for `torch.load` (CVE-2025-32434). Pinned to `transformers<4.51` to work around this. If upgrading Python to 3.10+, this constraint can be removed.
- Evaluation metrics: **Keyword Match Score** = percentage of expected keywords found in Top-3 result texts; **Top Score** = combined score of the highest-ranked result.

## Differences from Zenn Article

| Component | Zenn Article | This Implementation |
|-----------|-------------|---------------------|
| Dense Model | ruri-v3-310m (768-dim) | all-MiniLM-L6-v2 (384-dim) default, ruri tested separately |
| Sparse Method | bge-m3 learned sparse | TF-IDF / Jaccard (default), bge-m3 learned sparse (`test_bgem3_kenpo.py`) |
| Evaluation Metrics | P@K, MRR, NDCG | Keyword Match Score, Top Score |
| Hardware | GPU (RTX 4090) | CPU (macOS) + GPU (Colab T4) |
