# RAG System Verification - Based on Zenn Article

このプロジェクトは、[Zennの記事](https://zenn.dev/fp16/articles/aa48dcae23974e)で紹介されているRAGシステムを検証した実装です。

## 概要

Dense Embedding と Sparse Embedding のハイブリッド検索を実装し、TMC4361Aのデータシート（PDF）に対して検索精度を評価しました。

## 実装内容

### 1. 環境構築

- Python 3.9.6
- 主要ライブラリ:
  - `sentence-transformers`: Dense embedding用
  - `chromadb`: ベクトルデータベース
  - `pymupdf`: PDF処理
  - `torch`, `transformers`: モデル実行環境

### 2. アーキテクチャ

```
PDFファイル
    ↓
テキスト抽出 & チャンク分割 (pdf_processor.py)
    ↓
Dense Embedding (all-MiniLM-L6-v2)
    +
Sparse Embedding (TF-IDFベース)
    ↓
ハイブリッド検索 (rag_system.py)
    ↓
検索結果
```

### 3. ハイブリッド検索の実装

Zenn記事で紹介されている重み付き統合アルゴリズムを実装:

```python
def weighted_fusion(dense_scores, sparse_scores, alpha):
    dense_norm = min_max_normalize(dense_scores)
    sparse_norm = min_max_normalize(sparse_scores)
    return alpha * dense_norm + (1.0 - alpha) * sparse_norm
```

- `alpha`: Dense embeddingの重み (0.0〜1.0)
- `1-alpha`: Sparse embeddingの重み

## ファイル構成

```
/Users/seigo/Desktop/working/RagTest/
├── README.md                    # このファイル
├── requirements.txt             # 依存ライブラリ
├── pdf_processor.py             # PDF処理モジュール
├── rag_system.py                # RAGシステム本体
├── main.py                      # メイン実行スクリプト
├── evaluation.py                # 評価スクリプト
├── results_summary.json         # 基本実行結果
├── evaluation_results.json      # 詳細評価結果
└── TMC4361A_datasheet_rev1.26_01.pdf  # テストデータ
```

## 実行方法

### 1. 仮想環境のセットアップ

```bash
python3 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 基本的なRAGシステムのテスト

```bash
python main.py
```

### 3. 詳細評価の実行

```bash
python evaluation.py
```

## 評価結果

PDFから1,148個のチャンク（平均543文字/チャンク）を生成し、5つのテストクエリで評価を実施。

### パフォーマンス比較

| Alpha | 検索時間 (秒) | キーワード適合率 | 平均スコア |
|-------|---------------|------------------|------------|
| 0.0   | 0.1827        | 0.488            | 1.0000     |
| 0.3   | 0.1576        | 0.510            | 0.7000     |
| 0.5   | 0.1619        | 0.594            | 0.5441     |
| 0.7   | 0.1624        | **0.631**        | 0.7264     |
| 1.0   | **0.0340**    | 0.614            | 0.5397     |

### 主要な発見

1. **検索速度**: Dense検索のみ（alpha=1.0）が圧倒的に高速（0.03秒）
   - ハイブリッド検索は約5倍の時間がかかる（Sparse検索の計算コスト）

2. **検索精度**: alpha=0.7（Dense重視のハイブリッド）が最良
   - キーワード適合率: 0.631
   - Zenn記事の推奨値と一致

3. **バランス**: alpha=0.5でDenseとSparseを均等に組み合わせると、検索精度が向上

## Zenn記事との比較

### 記事で使用されているモデル

- **Dense**: ruri-v3-310m（日本語特化）、bge-m3、OpenAI text-embedding-3-large
- **Sparse**: bge-m3のSparse出力（トークン重みベクトル）

### 本実装で使用したモデル

- **Dense**: all-MiniLM-L6-v2（軽量、英語特化、384次元）
- **Sparse**: TF-IDFベースの簡易実装

### 主な違い

1. **モデルの規模**
   - 記事: 大規模モデル（768〜3072次元）
   - 本実装: 軽量モデル（384次元）
   - 理由: ローカル環境での実行可能性を優先

2. **Sparse Embedding**
   - 記事: bge-m3の学習済みSparse出力
   - 本実装: TF-IDFベースの統計的手法
   - 理由: シンプルで理解しやすい実装

3. **評価データ**
   - 記事: 日本語データセット
   - 本実装: 英語技術ドキュメント（TMC4361A）

## 学んだこと

### 1. ハイブリッド検索の有効性

記事で指摘されていた通り、Dense + Sparse のハイブリッド検索により、以下が実現できました:

- **Dense**: 意味的な類似性を捉える
- **Sparse**: 固有名詞や専門用語を正確にマッチング
- **Hybrid**: 両者の長所を組み合わせて検索精度向上

### 2. Alpha パラメータの調整

- **alpha=0.7**: 記事の推奨値、本評価でも最良の結果
- **alpha=1.0**: 高速だが、固有名詞の検索が弱い
- **alpha=0.0**: キーワード検索のみ、意味的な理解がない

### 3. 実装の簡略化

記事の高度な実装（bge-m3のSparse出力）を、TF-IDFで簡易的に再現できることを確認。
教育目的やプロトタイプには十分な精度が得られる。

## 今後の改善案

1. **多言語モデルの導入**
   - `paraphrase-multilingual-MiniLM-L12-v2` などで日本語対応

2. **より高度なSparse Embedding**
   - bge-m3モデルの導入
   - BM25アルゴリズムの実装

3. **リランキング**
   - 記事で触れられているリランキング層の追加
   - Cross-Encoderモデルの活用

4. **チャンク戦略の最適化**
   - 現在: 固定長512文字 + オーバーラップ50文字
   - 改善: セマンティックチャンキング、文境界を考慮した分割

5. **評価指標の拡充**
   - P@K (Precision at K)
   - MRR (Mean Reciprocal Rank)
   - NDCG (Normalized Discounted Cumulative Gain)

## 参考資料

- [Zenn記事: RAG実装の比較評価](https://zenn.dev/fp16/articles/aa48dcae23974e)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)

## ライセンス

このプロジェクトは教育・検証目的で作成されました。
