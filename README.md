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
- **Backend**: Python, FastAPI
- **AI Framework**: LangChain
- **AI Services**: Azure AI Services
- **Vector Database**: Vector Embedding & Storage
- **Document Processing**: Agent & Tool 기반 검증

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
├── app/
│   ├── api/                 # API 라우터
│   ├── core/               # 핵심 설정
│   ├── models/             # 데이터 모델
│   ├── services/           # 비즈니스 로직
│   └── utils/              # 유틸리티
├── docs/                   # 문서
├── tests/                  # 테스트
└── requirements.txt        # 의존성
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
# 개발 서버 실행
uvicorn app.main:app --reload
```

## 🔄 워크플로우

1. **문서 업로드** → AI 검증 → 내용 보완
2. **보완된 지식** → 게시판 등록 + Vector Embedding
3. **사용자 질의** → RAG 검색 → 답변 생성

## 🛠️ 개발 현황

현재 프로젝트는 초기 개발 단계에 있으며, 핵심 아키텍처 설계 및 기본 구조를 구축 중입니다.

## 📄 라이센스

이 프로젝트는 교육 목적으로 개발되었습니다.