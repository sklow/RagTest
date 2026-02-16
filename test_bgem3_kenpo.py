"""
日本国憲法PDF × BGE-M3 でDense + Learned Sparse ハイブリッド検索テスト

BGE-M3のSparse出力（lexical weights）を使用することで、
従来のTF-IDF/Jaccard方式では不可能だった日本語Sparse検索を実現する。
"""
from pdf_processor import PDFProcessor
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import json
import time


def cosine_similarity_matrix(query_vec, doc_vecs):
    """クエリベクトルと全ドキュメントベクトルのcosine similarityを一括計算"""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return np.dot(doc_norms, query_norm)


def min_max_normalize(scores):
    """Min-Max正規化"""
    scores = np.array(scores)
    if len(scores) == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)


def main():
    print("=" * 80)
    print("RAG Test: 日本国憲法 × BGE-M3 (Dense + Learned Sparse)")
    print("=" * 80)

    PDF_PATH = "/Users/seigo/Desktop/working/RagTest/nihonkokukenpou.pdf"
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
    print(f"\n  First chunk preview: {chunks[0]['text'][:200]}...")

    # Step 2: BGE-M3 モデル初期化
    print("\n[Step 2] BGE-M3 Model Initialization")
    print("-" * 80)
    t0 = time.time()
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    init_time = time.time() - t0
    print(f"  Model loaded in {init_time:.1f}s")

    # Step 3: ドキュメントエンコード（Dense + Sparse 一括）
    print("\n[Step 3] Encoding documents (Dense + Sparse)")
    print("-" * 80)
    chunk_texts = [c['text'] for c in chunks]

    t0 = time.time()
    doc_outputs = model.encode(
        chunk_texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )
    encode_time = time.time() - t0
    print(f"  Encoded {len(chunks)} chunks in {encode_time:.1f}s")
    print(f"  Dense vector dim: {doc_outputs['dense_vecs'].shape[1]}")
    print(f"  Sparse vectors: {len(doc_outputs['lexical_weights'])} entries")

    # Sparse出力のサンプル表示
    sample_sparse = doc_outputs['lexical_weights'][0]
    # lexical_weightsはtoken_id -> weightのdict。convert_id_to_tokenで可読化
    try:
        token_samples = model.convert_id_to_token(doc_outputs['lexical_weights'][:1])
        if isinstance(token_samples, list):
            readable = token_samples[0]
        else:
            readable = token_samples
        top_tokens = sorted(readable.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        # convert_id_to_tokenが使えない場合はtoken_idのまま表示
        top_tokens = sorted(sample_sparse.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Sparse sample (chunk_0 top-10 tokens):")
    for token, weight in top_tokens:
        print(f"    '{token}': {weight:.4f}")

    doc_dense_vecs = doc_outputs['dense_vecs']  # (N, 1024)
    doc_sparse_vecs = doc_outputs['lexical_weights']  # list of dicts

    # Step 4: テストクエリ実行
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

    TOP_K = 3
    ALPHA = 0.7
    results_all = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print("=" * 80)

        # クエリエンコード
        query_output = model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        query_dense = query_output['dense_vecs'][0]  # (1024,)
        query_sparse = query_output['lexical_weights'][0]  # dict

        # --- Dense検索 ---
        dense_scores = cosine_similarity_matrix(query_dense, doc_dense_vecs)
        dense_top_indices = np.argsort(dense_scores)[::-1][:TOP_K]

        print("\n--- Dense Search (BGE-M3, 1024-dim) ---")
        for j, idx in enumerate(dense_top_indices, 1):
            print(f"\n  [Rank {j}] {chunks[idx]['id']}, Score: {dense_scores[idx]:.4f}")
            print(f"  Text: {chunk_texts[idx][:300]}")

        # --- Sparse検索 (Learned Lexical Weights) ---
        sparse_scores = []
        for doc_sparse in doc_sparse_vecs:
            score = model.compute_lexical_matching_score(query_sparse, doc_sparse)
            sparse_scores.append(score)
        sparse_scores = np.array(sparse_scores)
        sparse_top_indices = np.argsort(sparse_scores)[::-1][:TOP_K]

        print(f"\n--- Sparse Search (BGE-M3 Lexical Weights) ---")
        for j, idx in enumerate(sparse_top_indices, 1):
            print(f"\n  [Rank {j}] {chunks[idx]['id']}, Score: {sparse_scores[idx]:.4f}")
            print(f"  Text: {chunk_texts[idx][:300]}")

        # --- Hybrid検索 (alpha加重融合) ---
        # Top-K*2 候補を集める
        n_candidates = TOP_K * 2
        dense_candidate_indices = np.argsort(dense_scores)[::-1][:n_candidates]
        sparse_candidate_indices = np.argsort(sparse_scores)[::-1][:n_candidates]
        candidate_indices = list(set(dense_candidate_indices.tolist() + sparse_candidate_indices.tolist()))

        # 候補のスコアをMin-Max正規化して融合
        cand_dense = np.array([dense_scores[idx] for idx in candidate_indices])
        cand_sparse = np.array([sparse_scores[idx] for idx in candidate_indices])
        cand_dense_norm = min_max_normalize(cand_dense)
        cand_sparse_norm = min_max_normalize(cand_sparse)
        cand_hybrid = ALPHA * cand_dense_norm + (1 - ALPHA) * cand_sparse_norm

        # ソートしてTop-K
        hybrid_order = np.argsort(cand_hybrid)[::-1][:TOP_K]

        print(f"\n--- Hybrid Search (alpha={ALPHA}) ---")
        hybrid_details = []
        for j, rank in enumerate(hybrid_order, 1):
            idx = candidate_indices[rank]
            print(f"\n  [Rank {j}] {chunks[idx]['id']}")
            print(f"    Combined: {cand_hybrid[rank]:.4f}  Dense(norm): {cand_dense_norm[rank]:.4f}  Sparse(norm): {cand_sparse_norm[rank]:.4f}")
            print(f"    Dense(raw): {dense_scores[idx]:.4f}  Sparse(raw): {sparse_scores[idx]:.4f}")
            print(f"    Text: {chunk_texts[idx][:300]}")
            hybrid_details.append({
                "id": chunks[idx]['id'],
                "combined_score": float(cand_hybrid[rank]),
                "dense_norm": float(cand_dense_norm[rank]),
                "sparse_norm": float(cand_sparse_norm[rank]),
                "dense_raw": float(dense_scores[idx]),
                "sparse_raw": float(sparse_scores[idx]),
            })

        # クエリのSparseトークンを表示
        try:
            query_token_readable = model.convert_id_to_token([query_sparse])
            if isinstance(query_token_readable, list):
                query_token_readable = query_token_readable[0]
            sorted_tokens = sorted(query_token_readable.items(), key=lambda x: x[1], reverse=True)[:8]
        except Exception:
            sorted_tokens = sorted(query_sparse.items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"\n  Query sparse tokens: {sorted_tokens}")

        results_all.append({
            "query": query,
            "dense_top1_score": float(dense_scores[dense_top_indices[0]]),
            "sparse_top1_score": float(sparse_scores[sparse_top_indices[0]]),
            "hybrid_top1_score": float(cand_hybrid[hybrid_order[0]]),
            "dense_top1_id": chunks[dense_top_indices[0]]['id'],
            "sparse_top1_id": chunks[sparse_top_indices[0]]['id'],
            "hybrid_top1_id": hybrid_details[0]['id'],
            "hybrid_top3": hybrid_details,
        })

    # サマリー
    print("\n" + "=" * 80)
    print("Summary: BGE-M3 Dense vs Sparse vs Hybrid")
    print("=" * 80)
    print(f"{'Query':<30} {'Dense':>8} {'Sparse':>8} {'Hybrid':>8}  Dense==Sparse?")
    print("-" * 90)
    for r in results_all:
        same_top1 = "Yes" if r['dense_top1_id'] == r['sparse_top1_id'] else "No"
        print(f"  {r['query'][:28]:<30} {r['dense_top1_score']:>8.4f} {r['sparse_top1_score']:>8.4f} {r['hybrid_top1_score']:>8.4f}  {same_top1}")

    avg_dense = np.mean([r['dense_top1_score'] for r in results_all])
    avg_sparse = np.mean([r['sparse_top1_score'] for r in results_all])
    avg_hybrid = np.mean([r['hybrid_top1_score'] for r in results_all])
    print(f"\n  {'Average':<30} {avg_dense:>8.4f} {avg_sparse:>8.4f} {avg_hybrid:>8.4f}")

    # 結果保存
    output_path = "/Users/seigo/Desktop/working/RagTest/bgem3_kenpo_results.json"
    output_data = {
        "model": "BAAI/bge-m3",
        "document": "nihonkokukenpou.pdf",
        "alpha": ALPHA,
        "init_time_sec": round(init_time, 1),
        "encode_time_sec": round(encode_time, 1),
        "num_chunks": len(chunks),
        "dense_dim": int(doc_outputs['dense_vecs'].shape[1]),
        "avg_dense_top1": round(float(avg_dense), 4),
        "avg_sparse_top1": round(float(avg_sparse), 4),
        "avg_hybrid_top1": round(float(avg_hybrid), 4),
        "queries": results_all,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
