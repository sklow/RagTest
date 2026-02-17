"""
ruri-v3-310m インタラクティブ検索CUI
Dense検索 / Hybrid検索を切り替えて日本国憲法を検索
"""
import sys
import time
from pdf_processor import PDFProcessor
from rag_system import HybridRAGSystem

PDF_PATH = "/Users/seigo/Desktop/working/RagTest/nihonkokukenpou.pdf"
MODEL_NAME = "cl-nagoya/ruri-v3-310m"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def print_header():
    print("=" * 70)
    print("  ruri-v3-310m インタラクティブ検索")
    print("  対象: 日本国憲法")
    print("=" * 70)


def print_help():
    print()
    print("コマンド:")
    print("  (テキスト入力)    検索クエリとして実行")
    print("  /mode             検索モード切替 (dense / hybrid)")
    print("  /alpha <値>       alphaパラメータ変更 (0.0〜1.0, hybrid時)")
    print("  /top <数>         表示件数変更 (1〜10)")
    print("  /status           現在の設定を表示")
    print("  /help             このヘルプを表示")
    print("  /quit             終了")
    print()


def print_status(mode, alpha, top_k):
    print(f"\n  モード: {mode}")
    print(f"  alpha:  {alpha} (dense={alpha}, sparse={1-alpha})")
    print(f"  表示件数: {top_k}")
    print()


def print_dense_results(results, elapsed):
    print(f"\n  検索時間: {elapsed:.3f}s  ヒット: {len(results)}件")
    print("-" * 70)
    for i, (doc_id, score, text) in enumerate(results, 1):
        print(f"\n  [{i}] {doc_id}  score: {score:.4f}")
        print(f"  {text[:300]}")
    print()


def print_hybrid_results(results, elapsed):
    print(f"\n  検索時間: {elapsed:.3f}s  ヒット: {len(results)}件")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r['id']}  combined: {r['score']:.4f}"
              f"  dense: {r['dense_score']:.4f}"
              f"  sparse: {r['sparse_score']:.4f}")
        print(f"  {r['text'][:300]}")
    print()


def init_system():
    print("\n[初期化中] PDF読み込み...")
    processor = PDFProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = processor.process_pdf(PDF_PATH)
    print(f"  チャンク数: {len(chunks)}")

    print("\n[初期化中] ruri-v3-310m ロード & エンベディング...")
    rag = HybridRAGSystem(
        collection_name="interactive_ruri_kenpo",
        model_name=MODEL_NAME
    )
    rag.add_documents(chunks)
    print("  準備完了")
    return rag


def main():
    print_header()

    rag = init_system()

    mode = "hybrid"
    alpha = 0.7
    top_k = 3

    print_help()
    print_status(mode, alpha, top_k)

    while True:
        try:
            query = input(f"[{mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not query:
            continue

        if query == "/quit" or query == "/exit":
            print("終了します。")
            break

        if query == "/help":
            print_help()
            continue

        if query == "/status":
            print_status(mode, alpha, top_k)
            continue

        if query == "/mode":
            mode = "dense" if mode == "hybrid" else "hybrid"
            print(f"  モード変更: {mode}")
            continue

        if query.startswith("/alpha"):
            parts = query.split()
            if len(parts) == 2:
                try:
                    val = float(parts[1])
                    if 0.0 <= val <= 1.0:
                        alpha = val
                        print(f"  alpha変更: {alpha}")
                    else:
                        print("  エラー: 0.0〜1.0の範囲で指定してください")
                except ValueError:
                    print("  エラー: 数値を指定してください")
            else:
                print(f"  現在のalpha: {alpha}")
            continue

        if query.startswith("/top"):
            parts = query.split()
            if len(parts) == 2:
                try:
                    val = int(parts[1])
                    if 1 <= val <= 10:
                        top_k = val
                        print(f"  表示件数変更: {top_k}")
                    else:
                        print("  エラー: 1〜10の範囲で指定してください")
                except ValueError:
                    print("  エラー: 整数を指定してください")
            else:
                print(f"  現在の表示件数: {top_k}")
            continue

        if query.startswith("/"):
            print(f"  不明なコマンド: {query}  (/help でヘルプ表示)")
            continue

        # 検索実行
        start = time.time()
        if mode == "dense":
            results = rag.dense_search(query, top_k=top_k)
            elapsed = time.time() - start
            print_dense_results(results, elapsed)
        else:
            results = rag.hybrid_search(query, alpha=alpha, top_k=top_k)
            elapsed = time.time() - start
            print_hybrid_results(results, elapsed)


if __name__ == "__main__":
    main()
