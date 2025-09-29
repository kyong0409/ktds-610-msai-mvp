"""
LLM 및 애플리케이션 설정 관리
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# LLM 설정
LLM_CONFIG = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "azure": {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": "2023-12-01-preview",
        "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    }
}

# Vector DB 설정
VECTOR_DB_CONFIG = {
    "dimension": 1536,  # OpenAI embeddings dimension
    "index_type": "IndexFlatL2",
    "similarity_threshold": 0.8,
}

# Document Processing 설정
DOCUMENT_CONFIG = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "supported_formats": [".txt", ".pdf", ".docx", ".pptx"],
    "chunk_size": 1000,
    "chunk_overlap": 200,
}

# Streamlit 페이지 설정
PAGE_CONFIG = {
    "page_title": "AI Knowledge Management System",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# 채팅 설정
CHAT_CONFIG = {
    "max_history": 50,
    "system_message": "당신은 도움이 되는 AI 어시스턴트입니다. 저장된 지식을 바탕으로 정확하고 유용한 답변을 제공합니다.",
}

def get_config(section: str) -> Dict[str, Any]:
    """설정 섹션 반환"""
    configs = {
        "llm": LLM_CONFIG,
        "vector_db": VECTOR_DB_CONFIG,
        "document": DOCUMENT_CONFIG,
        "page": PAGE_CONFIG,
        "chat": CHAT_CONFIG,
    }
    return configs.get(section, {})