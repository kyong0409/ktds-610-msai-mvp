"""
RAGService 데이터베이스 초기화 스크립트
"""
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# 데이터 디렉토리 생성
data_dir = "./data"
chroma_dir = "./data/chroma_db"
sqlite_path = "./data/board.db"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(chroma_dir, exist_ok=True)

# SQLite DB 초기화
conn = sqlite3.connect(sqlite_path)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS board_posts (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        enhanced_doc_url TEXT,
        original_doc_url TEXT,
        author TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        views INTEGER DEFAULT 0,
        quality_score INTEGER DEFAULT 0,
        metadata TEXT
    )
""")

conn.commit()
conn.close()

print(f"[OK] 데이터베이스 초기화 완료")
print(f"ChromaDB 디렉토리: {os.path.abspath(chroma_dir)}")
print(f"SQLite DB 파일: {os.path.abspath(sqlite_path)}")