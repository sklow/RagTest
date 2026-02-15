"""
マルチモデル比較評価スクリプト
異なるSentenceTransformerモデルの性能を包括的に比較
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


# 評価対象モデル
MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "dim": 384,
        "params": "22M",
        "description": "軽量・高速。現在使用中のベースライン",
        "prefix": None,
    },
    {
        "name": "all-mpnet-base-v2",
        "dim": 768,
        "params": "110M",
        "description": "SentenceTransformers最高品質モデル",
        "prefix": None,
    },
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "params": "118M",
        "description": "多言語対応。50+言語サポート",
        "prefix": None,
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "params": "33M",
        "description": "BAAI小型モデル。query prefix推奨",
        "prefix": "Represent this sentence: ",
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "params": "110M",
        "description": "BAAI中型モデル。高精度",
        "prefix": "Represent this sentence: ",
    },
    {
        "name": "intfloat/e5-small-v2",
        "dim": 384,
        "params": "33M",
        "description": "Microsoft E5小型。query: prefix推奨",
        "prefix": "query: ",
    },
    {
        "name": "intfloat/e5-base-v2",
        "dim": 768,
        "params": "110M",
        "description": "Microsoft E5中型。高精度",
        "prefix": "query: ",
    },
    {
        "name": "thenlper/gte-small",
        "dim": 384,
        "params": "33M",
        "description": "Alibaba GTE小型。汎用性高い",
        "prefix": None,
    },
    {
        "name": "thenlper/gte-base",
        "dim": 768,
        "params": "110M",
        "description": "Alibaba GTE中型",
        "prefix": None,
    },
]

# Alpha値の設定
ALPHA_VALUES = [0.0, 0.5, 0.7, 1.0]

PDF_PATH = "/Users/seigo/Desktop/working/RagTest/TMC4361A_datasheet_rev1.26_01.pdf"


def evaluate_single_model(model_info, chunks, alpha_values=ALPHA_VALUES):
    """単一モデルの評価を実行"""
    model_name = model_info["name"]
    prefix = model_info.get("prefix")

    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"  Dimensions: {model_info['dim']}, Params: {model_info['params']}")
    print(f"  Description: {model_info['description']}")
    print(f"{'='*80}")

    # モデル初期化時間の計測
    init_start = time.time()
    try:
        rag = HybridRAGSystem(
            collection_name=f"eval_{model_name.replace('/', '_')}",
            model_name=model_name,
        )
    except Exception as e:
        print(f"  ERROR: Failed to initialize model: {e}")
        return None
    init_time = time.time() - init_start
    print(f"  Init time: {init_time:.2f}s")

    # ドキュメント追加時間の計測
    add_start = time.time()
    rag.add_documents(chunks)
    add_time = time.time() - add_start
    print(f"  Document indexing time: {add_time:.2f}s")

    # メモリ使用量の概算（モデルパラメータベース）
    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024

    model_results = {
        "model_name": model_name,
        "dim": model_info["dim"],
        "params": model_info["params"],
        "description": model_info["description"],
        "init_time": init_time,
        "indexing_time": add_time,
        "memory_mb": round(memory_mb, 1),
        "alpha_results": {},
    }

    for alpha in alpha_values:
        print(f"\n  --- Alpha = {alpha} ---")
        alpha_key = f"alpha_{alpha}"
        query_results = []

        for test in TEST_CASES:
            query = test["query"]
            if prefix:
                query_with_prefix = prefix + query
            else:
                query_with_prefix = query

            # 検索実行
            search_start = time.time()
            if alpha == 1.0:
                raw_results = rag.dense_search(query_with_prefix, top_k=5)
                formatted = [
                    {"id": r[0], "score": r[1], "text": r[2],
                     "dense_score": r[1], "sparse_score": 0.0}
                    for r in raw_results
                ]
            elif alpha == 0.0:
                formatted = rag.hybrid_search(query, alpha=0.0, top_k=5)
            else:
                formatted = rag.hybrid_search(query_with_prefix, alpha=alpha, top_k=5)
            search_time = time.time() - search_start

            # キーワードマッチスコア計算（Top-3）
            keyword_matches = []
            for result in formatted[:3]:
                text_lower = result["text"].lower()
                matches = sum(
                    1 for kw in test["expected_keywords"]
                    if kw.lower() in text_lower
                )
                keyword_matches.append(matches / len(test["expected_keywords"]))
            avg_keyword_score = (
                sum(keyword_matches) / len(keyword_matches) if keyword_matches else 0
            )

            top_score = formatted[0]["score"] if formatted else 0

            query_results.append({
                "query": test["query"],
                "category": test["category"],
                "difficulty": test["difficulty"],
                "expected_best_method": test["expected_best_method"],
                "search_time": round(search_time, 4),
                "keyword_match_score": round(avg_keyword_score, 4),
                "top_score": round(top_score, 4),
            })

        # Alpha単位の集計
        avg_search_time = np.mean([r["search_time"] for r in query_results])
        avg_keyword_match = np.mean([r["keyword_match_score"] for r in query_results])
        avg_top_score = np.mean([r["top_score"] for r in query_results])

        # カテゴリ別集計
        category_scores = defaultdict(list)
        for r in query_results:
            category_scores[r["category"]].append(r["keyword_match_score"])
        category_avg = {
            cat: round(np.mean(scores), 4) for cat, scores in category_scores.items()
        }

        model_results["alpha_results"][alpha_key] = {
            "avg_search_time": round(avg_search_time, 4),
            "avg_keyword_match": round(avg_keyword_match, 4),
            "avg_top_score": round(avg_top_score, 4),
            "category_scores": category_avg,
            "query_details": query_results,
        }

        print(f"    Avg search time: {avg_search_time:.4f}s")
        print(f"    Avg keyword match: {avg_keyword_match:.4f}")
        print(f"    Avg top score: {avg_top_score:.4f}")
        for cat, score in category_avg.items():
            print(f"    {cat}: {score:.4f}")

    return model_results


def run_full_evaluation():
    """全モデルの評価を実行"""
    print("=" * 80)
    print("MULTI-MODEL RAG EVALUATION")
    print(f"Models: {len(MODELS)}, Test cases: {len(TEST_CASES)}, Alpha values: {ALPHA_VALUES}")
    print("=" * 80)

    # PDF処理
    processor = PDFProcessor(chunk_size=512, chunk_overlap=50)
    chunks = processor.process_pdf(PDF_PATH)
    print(f"Loaded {len(chunks)} chunks")

    all_results = []
    failed_models = []

    for model_info in MODELS:
        try:
            result = evaluate_single_model(model_info, chunks)
            if result:
                all_results.append(result)
            else:
                failed_models.append(model_info["name"])
        except Exception as e:
            print(f"\nERROR evaluating {model_info['name']}: {e}")
            traceback.print_exc()
            failed_models.append(model_info["name"])

    # 結果の保存
    output = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_chunks": len(chunks),
        "num_test_cases": len(TEST_CASES),
        "alpha_values": ALPHA_VALUES,
        "failed_models": failed_models,
        "results": all_results,
    }

    output_path = "/Users/seigo/Desktop/working/RagTest/model_comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"Successfully evaluated: {len(all_results)} models")
    print(f"Failed: {len(failed_models)} models ({', '.join(failed_models)})")

    # サマリー表示
    print_summary(all_results)

    return output


def print_summary(results):
    """結果サマリーを表示"""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY (alpha=0.7, Dense-focused hybrid)")
    print(f"{'='*80}")

    header = f"{'Model':<40} {'Dim':>4} {'Init(s)':>8} {'KeyMatch':>9} {'TopScore':>9}"
    print(header)
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["alpha_results"].get("alpha_0.7", {}).get("avg_keyword_match", 0), reverse=True):
        alpha_r = r["alpha_results"].get("alpha_0.7", r["alpha_results"].get("alpha_0.5", {}))
        print(
            f"{r['model_name']:<40} {r['dim']:>4} "
            f"{r['init_time']:>8.2f} "
            f"{alpha_r.get('avg_keyword_match', 0):>9.4f} "
            f"{alpha_r.get('avg_top_score', 0):>9.4f}"
        )

    # カテゴリ別ベストモデル
    print(f"\n{'='*80}")
    print("BEST MODEL BY CATEGORY (alpha=0.7)")
    print(f"{'='*80}")

    categories = set()
    for r in results:
        alpha_r = r["alpha_results"].get("alpha_0.7", {})
        categories.update(alpha_r.get("category_scores", {}).keys())

    for cat in sorted(categories):
        best_model = None
        best_score = -1
        for r in results:
            alpha_r = r["alpha_results"].get("alpha_0.7", {})
            score = alpha_r.get("category_scores", {}).get(cat, 0)
            if score > best_score:
                best_score = score
                best_model = r["model_name"]
        print(f"  {cat:<25} -> {best_model} ({best_score:.4f})")


if __name__ == "__main__":
    run_full_evaluation()
