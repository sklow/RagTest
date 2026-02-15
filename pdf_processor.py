"""
PDFからテキストを抽出し、チャンクに分割するモジュール
"""
import fitz  # PyMuPDF
from typing import List, Dict
import re


class PDFProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: チャンクの文字数
            chunk_overlap: チャンク間のオーバーラップ文字数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDFからテキストを抽出"""
        doc = fitz.open(pdf_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()

        doc.close()
        return text

    def clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        # 複数の改行を1つに
        text = re.sub(r'\n+', '\n', text)
        # 複数の空白を1つに
        text = re.sub(r' +', ' ', text)
        # 前後の空白を削除
        text = text.strip()
        return text

    def split_into_chunks(self, text: str) -> List[Dict[str, str]]:
        """テキストをチャンクに分割"""
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0

        while start < text_length:
            end = start + self.chunk_size

            # チャンクの終わりが文の途中の場合、次の改行まで延長
            if end < text_length:
                next_newline = text.find('\n', end)
                if next_newline != -1 and next_newline - end < 100:
                    end = next_newline

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    'id': f'chunk_{chunk_id}',
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1

            # オーバーラップを考慮して次のスタート位置を決定
            start = end - self.chunk_overlap

        return chunks

    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """PDFを処理してチャンクのリストを返す"""
        print(f"Processing PDF: {pdf_path}")

        # テキスト抽出
        raw_text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(raw_text)} characters")

        # クリーニング
        cleaned_text = self.clean_text(raw_text)
        print(f"Cleaned text: {len(cleaned_text)} characters")

        # チャンク分割
        chunks = self.split_into_chunks(cleaned_text)
        print(f"Created {len(chunks)} chunks")

        return chunks


if __name__ == "__main__":
    # テスト実行
    processor = PDFProcessor(chunk_size=512, chunk_overlap=50)
    pdf_path = "/Users/seigo/Desktop/working/RagTest/TMC4361A_datasheet_rev1.26_01.pdf"

    chunks = processor.process_pdf(pdf_path)

    # 最初のいくつかのチャンクを表示
    print("\n=== First 3 chunks ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i} (ID: {chunk['id']}):")
        print(f"Text preview: {chunk['text'][:200]}...")
        print(f"Length: {len(chunk['text'])} characters")
