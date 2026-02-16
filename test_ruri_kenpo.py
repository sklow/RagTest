"""
日本国憲法PDF × ruri-v3-310m でRAGテスト
"""
from pdf_processor import PDFProcessor
from rag_system import HybridRAGSystem
import json

# ruri-v3-310mはクエリに "検索クエリ: " プレフィックスが必要な場合がある
# SentenceTransformerが自動で処理する場合もあるので、まずそのままテスト
QUERY_PREFIX = ""


def main():
    print("=" * 80)
    print("RAG Test: 日本国憲法 × ruri-v3-310m")
    print("=" * 80)

    PDF_PATH = "/Users/seigo/Desktop/working/RagTest/nihonkokukenpou.pdf"
    MODEL_NAME = "cl-nagoya/ruri-v3-310m"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # Step 1: PDF処理
    print("\n[Step 1] PDF Processing")
    print("-" * 80)
    processor = PDFProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = processor.process_pdf(PDF_PATH)

    print(f"\nChunk Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average chunk length: {sum(len(c['text']) for c in chunks) / len(chunks):.1f} chars")
    print(f"  Min chunk length: {min(len(c['text']) for c in chunks)} chars")
    print(f"  Max chunk length: {max(len(c['text']) for c in chunks)} chars")

    # 最初のチャンクを確認
    print(f"\n  First chunk preview: {chunks[0]['text'][:200]}...")

    # Step 2: RAGシステム初期化（ruri-v3-310m）
    print("\n[Step 2] RAG System Initialization")
    print("-" * 80)
    rag = HybridRAGSystem(
        collection_name="nihonkoku_kenpou",
        model_name=MODEL_NAME
    )

    print("\n[Step 3] Adding documents to vector database")
    print("-" * 80)
    rag.add_documents(chunks)

    # Step 3: テストクエリ
    print("\n[Step 4] Testing RAG System")
    print("=" * 80)

    test_queries = [
        # 基本的な権利
        "国民の基本的人権について",
        "表現の自由とは何か",
        "生存権の保障について",
        # 統治機構
        "内閣総理大臣の権限",
        "国会の役割と権限",
        "裁判所の独立性",
        # 具体的な条文
        "天皇の地位と役割",
        "戦争の放棄について",
        "教育を受ける権利",
        # 意味的検索（間接的な表現）
        "国民が自由に意見を述べる権利",
        "健康で文化的な生活",
    ]

    results_all = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print("=" * 80)

        search_query = QUERY_PREFIX + query

        # Dense検索
        print("\n--- Dense Search (ruri-v3-310m) ---")
        dense_results = rag.dense_search(search_query, top_k=3)
        for j, (doc_id, score, text) in enumerate(dense_results, 1):
            print(f"\n  [Rank {j}] ID: {doc_id}, Score: {score:.4f}")
            print(f"  Text: {text[:300]}")

        # Hybrid検索（alpha=0.7）
        print("\n--- Hybrid Search (alpha=0.7, Dense-focused) ---")
        hybrid_results = rag.hybrid_search(search_query, alpha=0.7, top_k=3)
        for j, result in enumerate(hybrid_results, 1):
            print(f"\n  [Rank {j}] ID: {result['id']}")
            print(f"    Combined: {result['score']:.4f}  Dense: {result['dense_score']:.4f}  Sparse: {result['sparse_score']:.4f}")
            print(f"    Text: {result['text'][:300]}")

        results_all.append({
            "query": query,
            "dense_top1_score": dense_results[0][1] if dense_results else 0,
            "dense_top1_text": dense_results[0][2][:200] if dense_results else "",
            "hybrid_top1_score": hybrid_results[0]['score'] if hybrid_results else 0,
        })

    # 結果保存
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for r in results_all:
        print(f"  Q: {r['query']}")
        print(f"    Dense Top1: {r['dense_top1_score']:.4f}  Hybrid Top1: {r['hybrid_top1_score']:.4f}")

    output_path = "/Users/seigo/Desktop/working/RagTest/ruri_kenpo_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
