"""
マルチモデル比較評価スクリプト（高速版）
キャッシュ済みモデル + 小型モデルに限定
"""
import json
import time
import traceback
import sys
import os
import numpy as np
from collections import defaultdict

from pdf_processor import PDFProcessor
from rag_system import HybridRAGSystem
from test_cases import TEST_CASES

# 評価対象モデル（キャッシュ済み + 小型）
MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "params": "22M",
        "type": "small",
        "description": "軽量・高速。現在使用中のベースライン",
        "prefix": None,
    },
    {
        "name": "all-mpnet-base-v2",
        "dim": 768,
        "params": "110M",
        "type": "base",
        "description": "SentenceTransformers最高品質モデル",
        "prefix": None,
    },
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "params": "118M",
        "type": "multilingual",
        "description": "多言語対応。50+言語サポート",
        "prefix": None,
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "params": "33M",
        "type": "small",
        "description": "BAAI小型モデル。query prefix推奨",
        "prefix": "Represent this sentence: ",
    },
    {
        "name": "intfloat/e5-small-v2",
        "dim": 384,
        "params": "33M",
        "type": "small",
        "description": "Microsoft E5小型。query: prefix推奨",
        "prefix": "query: ",
    },
    {
        "name": "thenlper/gte-small",
        "dim": 384,
        "params": "33M",
        "type": "small",
        "description": "Alibaba GTE小型。汎用性高い",
        "prefix": None,
    },
]

ALPHA_VALUES = [0.0, 0.5, 0.7, 1.0]
PDF_PATH = "/Users/seigo/Desktop/working/RagTest/TMC4361A_datasheet_rev1.26_01.pdf"

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'


def evaluate_single_model(model_info, chunks):
    """単一モデルの評価を実行"""
    model_name = model_info["name"]
    prefix = model_info.get("prefix")

    print(f"\n{'='*80}", flush=True)
    print(f"Evaluating: {model_name}", flush=True)
    print(f"  Dimensions: {model_info['dim']}, Params: {model_info['params']}", flush=True)
    print(f"{'='*80}", flush=True)

    init_start = time.time()
    try:
        rag = HybridRAGSystem(
            collection_name=f"eval_{model_name.replace('/', '_').replace('-','_')}",
            model_name=model_name,
        )
    except Exception as e:
        print(f"  ERROR: Failed to initialize: {e}", flush=True)
        return None
    init_time = time.time() - init_start
    print(f"  Init time: {init_time:.2f}s", flush=True)

    add_start = time.time()
    rag.add_documents(chunks)
    add_time = time.time() - add_start
    print(f"  Indexing time: {add_time:.2f}s", flush=True)

    try:
        import psutil
        memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except:
        memory_mb = 0

    model_results = {
        "model_name": model_name,
        "dim": model_info["dim"],
        "params": model_info["params"],
        "type": model_info["type"],
        "description": model_info["description"],
        "init_time": round(init_time, 2),
        "indexing_time": round(add_time, 2),
        "memory_mb": round(memory_mb, 1),
        "alpha_results": {},
    }

    for alpha in ALPHA_VALUES:
        print(f"\n  Alpha = {alpha}", flush=True)
        query_results = []

        for test in TEST_CASES:
            query = test["query"]
            q_for_dense = (prefix + query) if prefix else query

            search_start = time.time()
            if alpha == 1.0:
                raw = rag.dense_search(q_for_dense, top_k=5)
                formatted = [
                    {"id": r[0], "score": r[1], "text": r[2],
                     "dense_score": r[1], "sparse_score": 0.0}
                    for r in raw
                ]
            elif alpha == 0.0:
                formatted = rag.hybrid_search(query, alpha=0.0, top_k=5)
            else:
                formatted = rag.hybrid_search(q_for_dense, alpha=alpha, top_k=5)
            search_time = time.time() - search_start

            keyword_matches = []
            for result in formatted[:3]:
                text_lower = result["text"].lower()
                matches = sum(
                    1 for kw in test["expected_keywords"]
                    if kw.lower() in text_lower
                )
                keyword_matches.append(matches / len(test["expected_keywords"]))
            avg_kw = sum(keyword_matches) / len(keyword_matches) if keyword_matches else 0
            top_score = formatted[0]["score"] if formatted else 0

            query_results.append({
                "query": test["query"],
                "category": test["category"],
                "difficulty": test["difficulty"],
                "search_time": round(search_time, 4),
                "keyword_match_score": round(avg_kw, 4),
                "top_score": round(top_score, 4),
                "top_text_preview": formatted[0]["text"][:100] if formatted else "",
            })

        avg_st = np.mean([r["search_time"] for r in query_results])
        avg_kw = np.mean([r["keyword_match_score"] for r in query_results])
        avg_ts = np.mean([r["top_score"] for r in query_results])

        cat_scores = defaultdict(list)
        for r in query_results:
            cat_scores[r["category"]].append(r["keyword_match_score"])
        cat_avg = {c: round(np.mean(s), 4) for c, s in cat_scores.items()}

        model_results["alpha_results"][f"alpha_{alpha}"] = {
            "avg_search_time": round(avg_st, 4),
            "avg_keyword_match": round(avg_kw, 4),
            "avg_top_score": round(avg_ts, 4),
            "category_scores": cat_avg,
            "query_details": query_results,
        }

        print(f"    KeyMatch: {avg_kw:.4f}, TopScore: {avg_ts:.4f}, Time: {avg_st:.4f}s", flush=True)

    return model_results


def run_evaluation():
    print("=" * 80, flush=True)
    print(f"MULTI-MODEL RAG EVALUATION ({len(MODELS)} models)", flush=True)
    print("=" * 80, flush=True)

    processor = PDFProcessor(chunk_size=512, chunk_overlap=50)
    chunks = processor.process_pdf(PDF_PATH)
    print(f"Loaded {len(chunks)} chunks\n", flush=True)

    all_results = []
    failed = []

    for mi in MODELS:
        try:
            r = evaluate_single_model(mi, chunks)
            if r:
                all_results.append(r)
            else:
                failed.append(mi["name"])
        except Exception as e:
            print(f"\nERROR: {mi['name']}: {e}", flush=True)
            traceback.print_exc()
            failed.append(mi["name"])

    output = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_chunks": len(chunks),
        "num_test_cases": len(TEST_CASES),
        "alpha_values": ALPHA_VALUES,
        "failed_models": failed,
        "results": all_results,
    }

    path = "/Users/seigo/Desktop/working/RagTest/model_comparison_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS SAVED: {path}", flush=True)
    print(f"Evaluated: {len(all_results)}, Failed: {len(failed)}", flush=True)

    # Summary table
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY (alpha=0.7)", flush=True)
    print(f"{'Model':<45} {'Dim':>4} {'KW':>7} {'Top':>7} {'Time':>7}", flush=True)
    print("-" * 80, flush=True)
    for r in sorted(all_results,
                    key=lambda x: x["alpha_results"].get("alpha_0.7", {}).get("avg_keyword_match", 0),
                    reverse=True):
        a = r["alpha_results"].get("alpha_0.7", {})
        print(f"{r['model_name']:<45} {r['dim']:>4} "
              f"{a.get('avg_keyword_match',0):>7.4f} "
              f"{a.get('avg_top_score',0):>7.4f} "
              f"{a.get('avg_search_time',0):>7.4f}", flush=True)

    # Category best
    print(f"\nBEST BY CATEGORY (alpha=0.7):", flush=True)
    cats = set()
    for r in all_results:
        cats.update(r["alpha_results"].get("alpha_0.7", {}).get("category_scores", {}).keys())
    for cat in sorted(cats):
        best_name, best_score = "", -1
        for r in all_results:
            s = r["alpha_results"].get("alpha_0.7", {}).get("category_scores", {}).get(cat, 0)
            if s > best_score:
                best_score = s
                best_name = r["model_name"]
        print(f"  {cat:<25} -> {best_name} ({best_score:.4f})", flush=True)


if __name__ == "__main__":
    run_evaluation()
