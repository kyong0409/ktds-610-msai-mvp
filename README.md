# AI Knowledge Management System with RAG-based Chatbot

Python & FastAPI 기반의 AI 지식관리체계 및 RAG 기반 챗봇 서비스

## 📋 프로젝트 개요

AI와 RAG(Retrieval-Augmented Generation) 기술을 활용한 지식관리 시스템으로, 문서 업로드부터 AI 기반 지식 검증, 보완, 그리고 챗봇을 통한 질의응답까지 전체적인 지식 생명주기를 관리합니다.

## 🏗️ 시스템 구성

### 주요 컴포넌트
- **문서 업로드 시스템**: PDF, PPT, DOC 등 다양한 형식의 문서 업로드
- **지식 게시판**: 보완된 지식 열람 및 관리
- **RAG 챗봇**: 저장된 지식 기반 질의응답 서비스

### 기술 스택
- **Frontend**: Streamlit (Python Web Framework)
- **Backend Services**: Python, LangChain
- **AI Services**: OpenAI, Azure OpenAI
- **Vector Database**: FAISS
- **Document Processing**: PyPDF2, python-docx, python-pptx

## 🚀 핵심 기능

### 1. 지식 업로드 및 검증
- 다양한 문서 형식 지원 (PDF, PPT, DOC 등)
- AI Agent와 Tool을 활용한 자동 내용 검증
- 부족하거나 오류가 있는 내용 자동 보완
- 품질 검증된 지식 생성

### 2. 지식 관리 및 저장
- 보완된 지식의 게시판 자동 등록
- Vector Embedding 처리
- VectorDB에 지식 저장 및 인덱싱
- 지식 검색 및 관리 기능

### 3. RAG 기반 챗봇
- 저장된 지식 기반 질의응답
- 컨텍스트 인식 대화
- 정확한 정보 검색 및 답변 생성
- 실시간 지식 활용

## 📁 프로젝트 구조

```
ktds-610-msai-mvp/
├── chatbot/                # 챗봇 기능 모듈
│   ├── __init__.py
│   ├── ui.py              # 챗봇 UI 렌더링
│   ├── service.py         # 챗봇 서비스 로직
│   └── components.py      # 챗봇 UI 컴포넌트
├── knowledge/              # 지식 관리 모듈
│   ├── __init__.py
│   ├── ui.py              # 지식 등록 UI
│   ├── service.py         # 지식 관리 서비스
│   └── components.py      # 지식 관리 컴포넌트
├── board/                  # 게시판 모듈
│   ├── __init__.py
│   ├── ui.py              # 게시판 UI
│   ├── service.py         # 게시판 서비스
│   └── components.py      # 게시판 컴포넌트
├── config/                 # 설정 관리
│   ├── __init__.py
│   └── settings.py        # LLM, DB 등 설정
├── components/             # 공통 UI 컴포넌트
│   ├── __init__.py
│   ├── sidebar.py         # 사이드바
│   └── chat.py            # 채팅 컴포넌트
├── services/               # 공통 서비스
│   ├── __init__.py
│   ├── document_analyzer.py
│   ├── llm_service.py
│   └── vector_db.py
├── utils/                  # 유틸리티
│   ├── __init__.py
│   ├── file_processor.py
│   └── session_manager.py
├── app.py                  # 기존 단일 파일 앱
├── app_new.py             # 모듈화된 메인 앱
├── .env.example           # 환경변수 템플릿
└── requirements.txt       # 의존성
```

## ⚙️ 설치 및 실행

### 필수 요구사항
- Python 3.8+
- Azure AI Services 계정
- Vector Database

### 설치
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 Azure AI 키 및 설정 추가
```

### 실행
```bash
# 기존 단일 파일 버전 실행
streamlit run app.py

# 모듈화된 버전 실행 (권장)
streamlit run app_new.py

# 특정 포트로 실행
streamlit run app_new.py --server.port 8501
```

## 🔄 워크플로우

1. **문서 업로드** → AI 검증 → 내용 보완
2. **보완된 지식** → 게시판 등록 + Vector Embedding
3. **사용자 질의** → RAG 검색 → 답변 생성

## 🛠️ 개발 현황

현재 프로젝트는 초기 개발 단계에 있으며, 핵심 아키텍처 설계 및 기본 구조를 구축 중입니다.

## 📄 라이센스

이 프로젝트는 교육 목적으로 개발되었습니다.