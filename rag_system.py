"""
RAGシステムの実装
Dense + Sparse のハイブリッド検索を実装
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import math


class HybridRAGSystem:
    def __init__(self, collection_name: str = "rag_documents", model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = None, reset: bool = True):
        """
        Args:
            collection_name: ChromaDBのコレクション名
            model_name: Sentence Transformersのモデル名
                       - all-MiniLM-L6-v2: 軽量で高速（384次元）
                       - paraphrase-multilingual-MiniLM-L12-v2: 多言語対応
            persist_directory: 永続化ディレクトリ（Noneならインメモリ）
            reset: Trueならコレクションをリセット（既存データを破棄）
        """
        print(f"Initializing RAG system with model: {model_name}")

        # Dense Embedding用のモデル
        self.dense_model = SentenceTransformer(model_name)

        # ChromaDB クライアント
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))

        # コレクションの初期化
        if reset:
            try:
                self.client.delete_collection(collection_name)
            except:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Collection '{collection_name}' initialized (count: {self.collection.count()})")

    def compute_dense_embedding(self, text: str) -> List[float]:
        """Dense embeddingを計算"""
        embedding = self.dense_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def compute_sparse_embedding(self, text: str) -> Dict[str, float]:
        """
        簡易的なSparse embedding（BM25風のTF-IDF）
        記事ではbge-m3のSparse出力を使用していますが、
        ここではシンプルなTF-IDFベースの実装を使用
        """
        # トークン化（簡易版）
        tokens = text.lower().split()

        # TF計算
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # 正規化
        total_tokens = len(tokens)
        sparse_vec = {}
        for token, count in tf.items():
            sparse_vec[token] = count / total_tokens

        return sparse_vec

    def add_documents(self, chunks: List[Dict[str, str]]):
        """ドキュメントをベクトルDBに追加"""
        print(f"Adding {len(chunks)} documents to the database...")

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Dense embedding
            dense_emb = self.compute_dense_embedding(chunk['text'])

            # Sparse embedding（メタデータとして保存）
            sparse_emb = self.compute_sparse_embedding(chunk['text'])

            ids.append(chunk['id'])
            embeddings.append(dense_emb)
            documents.append(chunk['text'])
            metadatas.append({
                'start_pos': chunk['start_pos'],
                'end_pos': chunk['end_pos'],
                'sparse_tokens': len(sparse_emb),
                # Sparse vectorはメタデータとして保存（簡易版）
            })

        # バッチで追加
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Successfully added {len(chunks)} documents")

    @staticmethod
    def min_max_normalize(scores: List[float]) -> np.ndarray:
        """Min-Max正規化（記事の手法を参考）"""
        scores = np.array(scores)
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score == 0:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def dense_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        """Dense検索のみ"""
        query_embedding = self.compute_dense_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        formatted_results = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distance = results['distances'][0][i]
                document = results['documents'][0][i]
                # コサイン距離をスコアに変換（1 - distance）
                score = 1 - distance
                formatted_results.append((doc_id, score, document))

        return formatted_results

    def sparse_search(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Tuple[str, float]]:
        """Sparse検索（簡易BM25風）"""
        query_tokens = set(query.lower().split())

        scores = []
        for doc in documents:
            doc_tokens = doc['text'].lower().split()
            doc_token_set = set(doc_tokens)

            # Jaccard類似度的なスコア
            intersection = query_tokens & doc_token_set
            union = query_tokens | doc_token_set

            if len(union) > 0:
                score = len(intersection) / len(union)
            else:
                score = 0.0

            scores.append((doc['id'], score))

        # スコアでソート
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def hybrid_search(self, query: str, alpha: float = 0.5, top_k: int = 10) -> List[Dict]:
        """
        ハイブリッド検索（記事の手法を参考）

        Args:
            query: 検索クエリ
            alpha: Dense重み（0.0〜1.0）。Sparse重みは (1 - alpha)
            top_k: 返す結果数

        Returns:
            検索結果のリスト
        """
        # Dense検索
        dense_results = self.dense_search(query, top_k=top_k * 2)

        # 全ドキュメントを取得してSparse検索用に準備
        all_docs_result = self.collection.get()
        all_docs = [
            {'id': all_docs_result['ids'][i], 'text': all_docs_result['documents'][i]}
            for i in range(len(all_docs_result['ids']))
        ]

        # Sparse検索
        sparse_results = self.sparse_search(query, all_docs, top_k=top_k * 2)

        # スコアの統合
        doc_scores = {}

        # Dense スコアを集計
        dense_ids = [r[0] for r in dense_results]
        dense_scores = [r[1] for r in dense_results]
        dense_normalized = self.min_max_normalize(dense_scores)

        for i, doc_id in enumerate(dense_ids):
            doc_scores[doc_id] = {'dense': dense_normalized[i], 'sparse': 0.0}

        # Sparse スコアを集計
        sparse_ids = [r[0] for r in sparse_results]
        sparse_scores_raw = [r[1] for r in sparse_results]
        sparse_normalized = self.min_max_normalize(sparse_scores_raw)

        for i, doc_id in enumerate(sparse_ids):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'dense': 0.0, 'sparse': sparse_normalized[i]}
            else:
                doc_scores[doc_id]['sparse'] = sparse_normalized[i]

        # 重み付き統合（記事の手法）
        final_scores = []
        for doc_id, scores in doc_scores.items():
            combined_score = alpha * scores['dense'] + (1 - alpha) * scores['sparse']
            final_scores.append((doc_id, combined_score, scores['dense'], scores['sparse']))

        # スコアでソート
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # 結果をフォーマット
        results = []
        for doc_id, combined_score, dense_score, sparse_score in final_scores[:top_k]:
            # ドキュメント本文を取得
            doc_result = self.collection.get(ids=[doc_id])
            if doc_result['documents']:
                results.append({
                    'id': doc_id,
                    'score': combined_score,
                    'dense_score': dense_score,
                    'sparse_score': sparse_score,
                    'text': doc_result['documents'][0]
                })

        return results


if __name__ == "__main__":
    # テスト実行は別ファイルで行う
    print("RAG System module loaded successfully")
