"""
評価結果の可視化スクリプト（テキストベースのチャート）
"""
import json


def visualize_results():
    """評価結果をテキストベースで可視化"""

    # 結果の読み込み
    with open("/Users/seigo/Desktop/working/RagTest/evaluation_results.json", 'r') as f:
        results = json.load(f)

    print("=" * 80)
    print("RAG System Evaluation - Visual Summary")
    print("=" * 80)

    # 1. キーワード適合率の比較
    print("\n1. Keyword Match Score by Alpha Value")
    print("-" * 80)
    print("Alpha | Score | Chart")
    print("-" * 80)

    max_keyword_score = max(r['avg_keyword_match'] for r in results)

    for result in results:
        alpha = result['alpha']
        score = result['avg_keyword_match']
        bar_length = int((score / max_keyword_score) * 50)
        bar = '█' * bar_length
        print(f"{alpha:>5.1f} | {score:.3f} | {bar}")

    # 2. 検索速度の比較
    print("\n2. Average Search Time by Alpha Value")
    print("-" * 80)
    print("Alpha | Time(s) | Chart")
    print("-" * 80)

    max_time = max(r['avg_search_time'] for r in results)

    for result in results:
        alpha = result['alpha']
        time = result['avg_search_time']
        bar_length = int((time / max_time) * 50)
        bar = '█' * bar_length
        print(f"{alpha:>5.1f} | {time:>7.4f} | {bar}")

    # 3. トップスコアの比較
    print("\n3. Average Top Score by Alpha Value")
    print("-" * 80)
    print("Alpha | Score | Chart")
    print("-" * 80)

    max_top_score = max(r['avg_top_score'] for r in results)

    for result in results:
        alpha = result['alpha']
        score = result['avg_top_score']
        bar_length = int((score / max_top_score) * 50)
        bar = '█' * bar_length
        print(f"{alpha:>5.1f} | {score:.4f} | {bar}")

    # 4. クエリごとの詳細分析
    print("\n4. Query-by-Query Analysis")
    print("=" * 80)

    # Alpha=0.7の結果を詳しく見る（記事推奨値）
    alpha_07_result = next(r for r in results if r['alpha'] == 0.7)

    for i, query_result in enumerate(alpha_07_result['query_results'], 1):
        print(f"\nQuery {i}: {query_result['query']}")
        print(f"  Keyword Match: {query_result['keyword_match_score']:.3f}")
        print(f"  Search Time: {query_result['search_time']:.4f}s")
        print(f"  Top-3 Results:")

        for j, res in enumerate(query_result['results'][:3], 1):
            print(f"\n    [{j}] {res['id']}")
            print(f"        Combined: {res['score']:.4f} | "
                  f"Dense: {res['dense_score']:.4f} | "
                  f"Sparse: {res['sparse_score']:.4f}")
            print(f"        Text: {res['text_preview'][:100]}...")

    # 5. 推奨設定
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\nBased on the evaluation results:")
    print("\n1. For BEST ACCURACY (Keyword Matching):")
    best_accuracy = max(results, key=lambda x: x['avg_keyword_match'])
    print(f"   → Use alpha = {best_accuracy['alpha']}")
    print(f"   → Keyword match score: {best_accuracy['avg_keyword_match']:.3f}")
    print(f"   → Search time: {best_accuracy['avg_search_time']:.4f}s")

    print("\n2. For FASTEST SEARCH:")
    fastest = min(results, key=lambda x: x['avg_search_time'])
    print(f"   → Use alpha = {fastest['alpha']}")
    print(f"   → Search time: {fastest['avg_search_time']:.4f}s")
    print(f"   → Keyword match score: {fastest['avg_keyword_match']:.3f}")

    print("\n3. For BALANCED APPROACH (Zenn Article Recommendation):")
    print(f"   → Use alpha = 0.7 (Dense-focused hybrid)")
    balanced = next(r for r in results if r['alpha'] == 0.7)
    print(f"   → Keyword match score: {balanced['avg_keyword_match']:.3f}")
    print(f"   → Search time: {balanced['avg_search_time']:.4f}s")
    print(f"   → This balances accuracy and semantic understanding")

    print("\n4. Implementation Notes:")
    print("   - Alpha=1.0: Pure Dense (fast, semantic)")
    print("   - Alpha=0.7: Dense-focused hybrid (recommended)")
    print("   - Alpha=0.5: Balanced hybrid")
    print("   - Alpha=0.3: Sparse-focused hybrid")
    print("   - Alpha=0.0: Pure Sparse (keyword matching)")

    # 6. 統計サマリー
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    print(f"\nDataset Statistics:")
    with open("/Users/seigo/Desktop/working/RagTest/results_summary.json", 'r') as f:
        summary = json.load(f)

    print(f"  PDF: {summary['pdf_path'].split('/')[-1]}")
    print(f"  Total chunks: {summary['chunk_statistics']['total_chunks']}")
    print(f"  Chunk size: {summary['chunk_statistics']['chunk_size']} chars")
    print(f"  Chunk overlap: {summary['chunk_statistics']['chunk_overlap']} chars")
    print(f"  Average chunk length: {summary['chunk_statistics']['avg_length']:.1f} chars")
    print(f"  Model: {summary['model']}")

    print(f"\nEvaluation Setup:")
    print(f"  Number of test queries: {len(alpha_07_result['query_results'])}")
    print(f"  Alpha values tested: {[r['alpha'] for r in results]}")
    print(f"  Top-K results per query: 5")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    visualize_results()
