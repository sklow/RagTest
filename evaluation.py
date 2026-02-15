"""
RAGシステムの評価スクリプト
異なるalphaパラメータでの性能比較
"""
from pdf_processor import PDFProcessor
from rag_system import HybridRAGSystem
import json
import time


def evaluate_rag_system():
    """RAGシステムの詳細評価"""
    print("=" * 80)
    print("Detailed RAG System Evaluation")
    print("=" * 80)

    # PDFの処理
    PDF_PATH = "/Users/seigo/Desktop/working/RagTest/TMC4361A_datasheet_rev1.26_01.pdf"
    processor = PDFProcessor(chunk_size=512, chunk_overlap=50)
    chunks = processor.process_pdf(PDF_PATH)

    # RAGシステムの初期化
    rag = HybridRAGSystem(
        collection_name="tmc4361a_evaluation",
        model_name="all-MiniLM-L6-v2"
    )
    rag.add_documents(chunks)

    # 評価用クエリ
    test_queries = [
        {
            "query": "What is the maximum voltage?",
            "expected_keywords": ["voltage", "maximum", "VCC", "3.3V", "5.0V"]
        },
        {
            "query": "SPI interface configuration",
            "expected_keywords": ["SPI", "interface", "configuration", "communication"]
        },
        {
            "query": "Motor control operation",
            "expected_keywords": ["motor", "control", "operation", "encoder"]
        },
        {
            "query": "TMC4361A specifications",
            "expected_keywords": ["TMC4361A", "specification", "datasheet"]
        },
        {
            "query": "Closed-loop control",
            "expected_keywords": ["closed-loop", "PID", "feedback"]
        }
    ]

    # 異なるalphaパラメータでの評価
    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = []

    print("\n" + "=" * 80)
    print("Evaluating different alpha values (Dense vs Sparse weight)")
    print("=" * 80)

    for alpha in alpha_values:
        print(f"\n{'=' * 80}")
        print(f"Alpha = {alpha} (Dense: {alpha:.1f}, Sparse: {1-alpha:.1f})")
        print('=' * 80)

        query_results = []

        for i, test in enumerate(test_queries, 1):
            query = test["query"]
            expected_keywords = test["expected_keywords"]

            print(f"\nQuery {i}: {query}")

            # 検索実行
            start_time = time.time()
            if alpha == 1.0:
                # Dense検索のみ
                search_results = rag.dense_search(query, top_k=5)
                formatted_results = [
                    {
                        'id': r[0],
                        'score': r[1],
                        'text': r[2],
                        'dense_score': r[1],
                        'sparse_score': 0.0
                    }
                    for r in search_results
                ]
            elif alpha == 0.0:
                # Sparse検索のみ（ハイブリッドでalpha=0）
                formatted_results = rag.hybrid_search(query, alpha=0.0, top_k=5)
            else:
                # ハイブリッド検索
                formatted_results = rag.hybrid_search(query, alpha=alpha, top_k=5)

            search_time = time.time() - start_time

            # キーワードマッチングスコアの計算
            keyword_matches = []
            for result in formatted_results[:3]:  # Top-3での評価
                text_lower = result['text'].lower()
                matches = sum(1 for keyword in expected_keywords if keyword.lower() in text_lower)
                keyword_matches.append(matches / len(expected_keywords))

            avg_keyword_score = sum(keyword_matches) / len(keyword_matches) if keyword_matches else 0

            print(f"  Search time: {search_time:.4f}s")
            print(f"  Keyword match score (Top-3 avg): {avg_keyword_score:.3f}")
            print(f"  Top result score: {formatted_results[0]['score']:.4f}")

            query_results.append({
                'query': query,
                'search_time': search_time,
                'keyword_match_score': avg_keyword_score,
                'top_score': formatted_results[0]['score'],
                'results': [
                    {
                        'rank': i+1,
                        'id': r['id'],
                        'score': r['score'],
                        'dense_score': r['dense_score'],
                        'sparse_score': r['sparse_score'],
                        'text_preview': r['text'][:150]
                    }
                    for i, r in enumerate(formatted_results[:5])
                ]
            })

        # 平均スコアの計算
        avg_search_time = sum(r['search_time'] for r in query_results) / len(query_results)
        avg_keyword_match = sum(r['keyword_match_score'] for r in query_results) / len(query_results)
        avg_top_score = sum(r['top_score'] for r in query_results) / len(query_results)

        print(f"\n{'=' * 80}")
        print(f"Summary for alpha={alpha}")
        print(f"  Average search time: {avg_search_time:.4f}s")
        print(f"  Average keyword match score: {avg_keyword_match:.3f}")
        print(f"  Average top score: {avg_top_score:.4f}")

        results.append({
            'alpha': alpha,
            'avg_search_time': avg_search_time,
            'avg_keyword_match': avg_keyword_match,
            'avg_top_score': avg_top_score,
            'query_results': query_results
        })

    # 結果の保存
    print("\n" + "=" * 80)
    print("Saving evaluation results...")
    print("=" * 80)

    output_file = "/Users/seigo/Desktop/working/RagTest/evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")

    # サマリーの表示
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Alpha':<10} {'Avg Time (s)':<15} {'Keyword Match':<15} {'Avg Top Score':<15}")
    print("-" * 80)
    for result in results:
        print(f"{result['alpha']:<10.1f} {result['avg_search_time']:<15.4f} "
              f"{result['avg_keyword_match']:<15.3f} {result['avg_top_score']:<15.4f}")

    # ベストアルファの推奨
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_keyword_match = max(results, key=lambda x: x['avg_keyword_match'])
    best_top_score = max(results, key=lambda x: x['avg_top_score'])

    print(f"\nBest alpha for keyword matching: {best_keyword_match['alpha']} "
          f"(score: {best_keyword_match['avg_keyword_match']:.3f})")
    print(f"Best alpha for top score: {best_top_score['alpha']} "
          f"(score: {best_top_score['avg_top_score']:.4f})")

    print("\nInterpretation:")
    print("  - alpha=1.0: Pure Dense embedding search (semantic similarity)")
    print("  - alpha=0.0: Pure Sparse search (keyword/term matching)")
    print("  - alpha=0.5: Balanced hybrid approach")
    print("  - alpha=0.7: Dense-focused hybrid (recommended by Zenn article for general use)")

    return results


if __name__ == "__main__":
    evaluate_rag_system()
