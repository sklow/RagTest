"""
RAGシステムのメイン実行スクリプト
"""
from pdf_processor import PDFProcessor
from rag_system import HybridRAGSystem
import json


def main():
    print("=" * 80)
    print("RAG System Verification - Based on Zenn Article")
    print("=" * 80)

    # 設定
    PDF_PATH = "/Users/seigo/Desktop/working/RagTest/TMC4361A_datasheet_rev1.26_01.pdf"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # ステップ1: PDFからテキスト抽出とチャンク分割
    print("\n[Step 1] PDF Processing")
    print("-" * 80)
    processor = PDFProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = processor.process_pdf(PDF_PATH)

    # チャンクの統計情報
    print(f"\nChunk Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average chunk length: {sum(len(c['text']) for c in chunks) / len(chunks):.1f} chars")
    print(f"  Min chunk length: {min(len(c['text']) for c in chunks)} chars")
    print(f"  Max chunk length: {max(len(c['text']) for c in chunks)} chars")

    # ステップ2: RAGシステムの初期化とドキュメント追加
    print("\n[Step 2] RAG System Initialization")
    print("-" * 80)
    rag = HybridRAGSystem(
        collection_name="tmc4361a_datasheet",
        model_name="all-MiniLM-L6-v2"  # 軽量で高速なモデル
    )

    print("\n[Step 3] Adding documents to vector database")
    print("-" * 80)
    rag.add_documents(chunks)

    # ステップ3: テストクエリの実行
    print("\n[Step 4] Testing RAG System")
    print("=" * 80)

    test_queries = [
        "What is the maximum voltage?",
        "TMC4361A features and specifications",
        "SPI interface configuration",
        "Current control settings",
        "Motor control operation"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print('=' * 80)

        # Dense検索のみ
        print("\n--- Dense Search Only ---")
        dense_results = rag.dense_search(query, top_k=3)
        for j, (doc_id, score, text) in enumerate(dense_results, 1):
            print(f"\n[Rank {j}] ID: {doc_id}, Score: {score:.4f}")
            print(f"Text preview: {text[:200]}...")

        # ハイブリッド検索（alpha=0.5: Dense 50%, Sparse 50%）
        print("\n--- Hybrid Search (alpha=0.5) ---")
        hybrid_results = rag.hybrid_search(query, alpha=0.5, top_k=3)
        for j, result in enumerate(hybrid_results, 1):
            print(f"\n[Rank {j}] ID: {result['id']}")
            print(f"  Combined Score: {result['score']:.4f}")
            print(f"  Dense Score: {result['dense_score']:.4f}")
            print(f"  Sparse Score: {result['sparse_score']:.4f}")
            print(f"  Text preview: {result['text'][:200]}...")

        # ハイブリッド検索（alpha=0.7: Dense重視）
        print("\n--- Hybrid Search (alpha=0.7, Dense-focused) ---")
        hybrid_results_dense = rag.hybrid_search(query, alpha=0.7, top_k=3)
        for j, result in enumerate(hybrid_results_dense, 1):
            print(f"\n[Rank {j}] ID: {result['id']}")
            print(f"  Combined Score: {result['score']:.4f}")
            print(f"  Dense Score: {result['dense_score']:.4f}")
            print(f"  Sparse Score: {result['sparse_score']:.4f}")
            print(f"  Text preview: {result['text'][:200]}...")

    print("\n" + "=" * 80)
    print("RAG System Verification Complete")
    print("=" * 80)

    # 結果をファイルに保存
    print("\n[Step 5] Saving results to file")
    results_summary = {
        "pdf_path": PDF_PATH,
        "chunk_statistics": {
            "total_chunks": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "avg_length": sum(len(c['text']) for c in chunks) / len(chunks),
        },
        "model": "all-MiniLM-L6-v2",
        "test_queries": test_queries
    }

    with open("/Users/seigo/Desktop/working/RagTest/results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("Results saved to: /Users/seigo/Desktop/working/RagTest/results_summary.json")


if __name__ == "__main__":
    main()
