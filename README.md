# AI Knowledge Management System with Multi-Agent Knowledge Creation

## URL
http://wa-ktds610.koreacentral.cloudapp.azure.com:8501/

AI 기반 지식관리 시스템으로, 문서 분석부터 지식 보완, Multi-Agent 기반 지식 창출까지 전체 지식 생명주기를 관리합니다.

## 📋 프로젝트 개요

LangChain, LangGraph, Azure AI Services를 활용한 지능형 지식관리 플랫폼입니다. 문서 업로드, AI 기반 분석 및 보완, 그리고 Multi-Agent 시스템을 통한 새로운 지식 창출까지 지원합니다.

## 🏗️ 시스템 아키텍처

### 기술 스택
- **Frontend**: Streamlit (Python Web Framework)
- **LLM Framework**: LangChain, LangGraph
- **AI Services**: Azure OpenAI (GPT-4o, GPT-4.1-mini)
- **Vector Database**: ChromaDB
- **Document Storage**: Azure Blob Storage
- **Board Database**: SQLite
- **Document Processing**: MarkItDown, Azure Document Intelligence

### Azure 서비스 사용 현황

| 기능 | Azure 서비스 | 용도 |
|------|-------------|------|
| **지식 등록** | Azure OpenAI | 문서 분석 및 보완 내용 생성 |
| | Azure Document Intelligence | PDF/PPT/DOCX 파일 텍스트 추출 |
| | Azure Blob Storage | 원본/보완 문서 저장 |
| | Azure OpenAI Embeddings | 텍스트 벡터화 (text-embedding-3-large) |
| **게시판** | Azure Blob Storage | 문서 다운로드 링크 제공 |
| **지식 창출** | LangGraph | Multi-Agent 지식 융합 및 창출 |
| | Azure OpenAI Embeddings | 유사도 검색 및 샘플링 |
| | ChromaDB | 벡터 검색 및 RAG |

## 🚀 핵심 기능

### 1. 📚 지식 등록

문서를 업로드하고 AI가 자동으로 분석하여 보완된 지식 문서를 생성합니다.

#### 상세 프로세스

```
1. 문서 업로드 (TXT, PDF, DOCX, PPTX)
   ↓
2. MarkItDown + Azure Document Intelligence로 텍스트 변환
   ↓
3. Azure OpenAI (GPT-4.1-mini)로 문서 분석
   - 메타데이터 추출 (문서 종류, 주제, 작성자, 키워드)
   - 문서 구조 분석 (섹션별 요약)
   - 활용 가능성 평가
   - 보완 필요 사항 도출
   ↓
4. AI가 보완된 지식 문서 생성 (표준 형식)
   - 메타데이터
   - 본문 (목적, 배경, 접근방법, 결과, 한계, 적용방안, 참고자료)
   ↓
5. Azure Blob Storage에 원본/보완 문서 업로드
   - original 컨테이너: 원본 파일
   - enhanced 컨테이너: 보완된 마크다운 파일
   ↓
6. 벡터화 및 저장
   - Azure OpenAI Embeddings (text-embedding-3-large)로 임베딩
   - SemanticChunker로 의미 기반 청크 분할
   - ChromaDB에 벡터 저장
   ↓
7. 게시판 DB에 메타데이터 저장 (SQLite)
   - 제목, 내용, 작성자, 품질점수
   - 원본/보완 문서 다운로드 URL
```

**사용 Azure 서비스:**
- **Azure OpenAI**: 문서 분석, 보완 내용 생성
- **Azure Document Intelligence**: PDF/Office 파일 텍스트 추출
- **Azure Blob Storage**: 문서 저장 및 다운로드 링크 제공
- **Azure OpenAI Embeddings**: 텍스트 벡터화

### 2. 📋 게시판

보완된 지식 문서를 조회하고 다운로드할 수 있습니다.

#### 상세 프로세스

```
1. SQLite DB에서 게시글 목록 조회
   ↓
2. 게시글 표시 (제목, 작성자, 품질점수, 조회수)
   ↓
3. 게시글 상세보기
   - 보완된 문서 내용 표시
   - Azure Blob Storage URL로 원본/보완 문서 다운로드 링크 제공
   ↓
4. 조회수 자동 증가 (SQLite DB 업데이트)
```

**사용 Azure 서비스:**
- **Azure Blob Storage**: 저장된 문서 다운로드

### 3. 🔬 지식 창출

Multi-Agent 시스템이 기존 지식을 분석하여 새로운 지식(K-Note)을 창출합니다.

#### 상세 프로세스 (LangGraph 8단계 워크플로우)

#### **1단계: 데이터 정규화 (Librarian Agent)**
```
목적: 품질이 검증된 지식 청크만 수집
```
- **입력**: ChromaDB 전체 벡터 데이터
- **처리**:
  - `quality_score ≥ 60%` 필터링
  - `pii_flag == False` (개인정보 제외)
  - Prefetch 최대 5000개 → Prekeep 1200개로 샘플링
- **출력**: `all_chunks_meta[]` (품질 검증된 청크 메타데이터)
- **기술**: ChromaDB 메타데이터 쿼리, NumPy 랜덤 샘플링

---

#### **2단계: 다양성 샘플링 (Sampler Agent)**
```
목적: 품질·다양성·최신성을 모두 고려한 최적 샘플 선택
```
- **입력**: 1단계 필터링된 청크 풀
- **알고리즘**: MMR (Maximal Marginal Relevance)
  - **품질 점수** (quality): Z-정규화
  - **최신성 점수** (recency): 날짜 기반 계산 (최근일수록 높음)
  - **다양성 점수** (diversity): 코사인 유사도 기반 (1 - max_similarity)
  - **종합 점수**: `λ * diversity + (1-λ) * (0.5*quality + 0.5*recency)`
- **제약 조건**:
  - 도메인당 최대 2개 (`max_per_domain=2`)
  - 평균 유사도 < 0.30 (`max_avg_sim=0.30`)
  - 목표 샘플 수: 10개 (`k=10`)
- **출력**: `samples[]` (최적 다양성 샘플 10개)
- **기술**: 코사인 유사도, Z-정규화, 탐욕적 선택

---

#### **3단계: 구조화 요약 (Summarizer Agent)**
```
목적: 샘플을 LLM이 처리 가능한 구조화된 형식으로 변환
모델: GPT-4.1-mini (Temperature: 0.0)
```
- **입력**: 2단계 샘플 청크 텍스트 (각 1000자)
- **LLM 프롬프트**:
  ```
  텍스트만 근거로 기술 문서 구조 요약(JSON) 생성.
  필드: problem, constraints[], approach, outcomes[], signals[], risks[]
  ```
- **처리**: 각 샘플마다 LLM 호출 → JSON 파싱
- **출력**: `summaries[]` (구조화된 요약 10개)
  ```json
  {
    "doc_id": "...",
    "chunk_id": "...",
    "problem": "해결하려는 문제",
    "constraints": ["제약1", "제약2"],
    "approach": "접근 방법",
    "outcomes": ["결과1", "결과2"],
    "signals": ["긍정 신호1", "신호2"],
    "risks": ["위험1", "위험2"]
  }
  ```
- **기술**: Azure OpenAI GPT-4.1-mini, JSON 파싱

---

#### **4단계: RAG 컨텍스트 확장 (Expander Agent)**
```
목적: 각 요약에 관련된 추가 지식을 ChromaDB에서 검색
```
- **입력**: 3단계 요약 (problem, approach, constraints)
- **처리**:
  - 각 요약에서 검색 쿼리 생성: `[problem, approach, constraint1, constraint2]`
  - ChromaDB 유사도 검색 (`top_k=6`)
  - 요약당 최대 6개 관련 청크 수집
- **출력**: `expansions{}` (요약 ID → 관련 청크 매핑)
  ```python
  {
    "doc1::chunk1": [관련청크1, 관련청크2, ...],
    "doc2::chunk2": [관련청크1, 관련청크2, ...]
  }
  ```
- **기술**: ChromaDB 벡터 유사도 검색 (cosine similarity)

---

#### **5단계: 지식 융합 (Synthesizer Agent)** ⭐ **핵심**
```
목적: 두 개의 다른 지식을 융합하여 새로운 통찰 창출
모델: GPT-4o (Temperature: 0.3, 창의성 활성화)
```
- **입력**: 요약 페어 (A, B) + 각각의 RAG 컨텍스트
- **LLM 프롬프트**:
  ```
  [역할] 아날로지 기반 설계자
  A={요약A}, B={요약B}
  A_ctx={RAG컨텍스트A}, B_ctx={RAG컨텍스트B}

  [지시] Analogical/Pattern/Bridging 제안 1~2개 생성
  - statement: 제안 명제
  - applicability: when/when_not/assumptions
  - expected_effects: 예상 효과
  - risks_limits: 위험 및 한계
  - evidence: [doc_id, chunk_id, quote, confidence]
  - quick_experiment: 빠른 검증 실험
  ```
- **처리**:
  - 요약을 짝수/홀수로 페어링 (총 5쌍)
  - 각 페어마다 LLM 호출 → 아날로지 기반 제안 생성
- **출력**: `proposals[]` (5~10개 융합 제안)
  ```json
  {
    "kind": "analogical | pattern | bridging",
    "statement": "A의 패턴을 B에 적용하면...",
    "applicability": {
      "when": ["조건1", "조건2"],
      "when_not": ["비적용 조건"],
      "assumptions": ["전제1", "전제2"]
    },
    "expected_effects": {"latency": "-20%", "throughput": "+30%"},
    "risks_limits": ["위험1", "한계1"],
    "evidence": [{"doc_id": "...", "confidence": 0.85}],
    "quick_experiment": {"setup": "...", "measure": "..."}
  }
  ```
- **기술**: GPT-4o, 아날로지 추론, 크로스 도메인 지식 융합

---

#### **6단계: 검증 (Verifier Agent)**
```
목적: 제안의 타당성 검증 및 반례 검토
모델: GPT-4.1-mini (Temperature: 0.0)
```
- **입력**: 5단계 제안 + ChromaDB 반례 검색
- **처리**:
  - 제안의 statement + assumptions로 ChromaDB 검색 (`top_k=3`)
  - 반례 증거 수집
  - LLM 프롬프트:
    ```
    [역할] 검증자
    proposal={제안}
    counter_evidence={반례}

    [지시] 반례/편향/외삽 위험 평가
    - verdict: accept | revise | reject
    - reasons[], added_evidence[]
    ```
- **출력**: `verdicts[]` (제안별 판정)
  ```json
  {
    "verdict": "accept",
    "reasons": ["근거가 충분함", "반례 없음"],
    "added_evidence": [{"source": "...", "text": "..."}]
  }
  ```
- **기술**: GPT-4.1-mini, 반례 검색, 비판적 평가
- **참고**: `enable_verification=False` 설정 시 자동 승인

---

#### **7단계: K-Note 생성 (Productizer Agent)**
```
목적: 승인된 제안을 표준 K-Note 형식으로 문서화
모델: GPT-4.1-mini (Temperature: 0.1)
```
- **입력**: 6단계에서 `verdict=accept`인 제안만
- **LLM 프롬프트**:
  ```
  proposal을 K-Note 스키마로 변환해 JSON만 출력
  필수: k_note_id, title, proposal, applicability, evidence,
        metrics_effect, risks_limits, recommended_experiments,
        status, owners, version, related
  ```
- **처리**: 승인된 제안 → K-Note 표준 형식 변환
- **출력**: `knotes[]` (K-Note 문서)
  ```json
  {
    "k_note_id": "KN-a3f8b2c1",
    "title": "CQRS 패턴의 Event Sourcing 적용",
    "proposal": "...",
    "applicability": {"when": [...], "when_not": [...], "assumptions": [...]},
    "evidence": [{"doc_id": "...", "chunk_id": "...", "quote": "...", "confidence": 0.85}],
    "metrics_effect": {"latency": "-20%", "throughput": "+30%"},
    "risks_limits": ["복잡도 증가", "학습 곡선"],
    "recommended_experiments": [{"setup": "...", "measure": "...", "success_criteria": "..."}],
    "status": "validated",
    "owners": ["Architecture Team"],
    "version": "1.0",
    "related": ["KN-xyz123"]
  }
  ```
- **기술**: GPT-4.1-mini, 구조화된 문서 생성

---

#### **8단계: 품질 평가 (Evaluator Agent)**
```
목적: K-Note 품질 평가 및 반복 여부 결정
```
- **입력**: 7단계 K-Note
- **평가 지표**:
  - **신규성 (novelty)**: 기존 지식 대비 새로운 정도 (현재: 랜덤 0.6~0.9)
  - **커버리지 (coverage)**: 문제 영역 포괄 범위 (현재: 랜덤 0.6~0.9)
  - **유용성 (utility)**: 실무 적용 가능성 (현재: 랜덤 0.6~0.9)
- **반복 로직**:
  - 평균 점수 ≥ 품질 임계값 (기본 0.75) → 종료
  - 평균 점수 < 임계값 → 2단계부터 재실행 (최대 3회)
- **출력**: `scores{}` (품질 점수)
- **참고**: ⚠️ 현재 랜덤 점수 사용 중 (개선 필요)

---

#### **구체화 단계: K-Note → 표준 지식 문서**
```
목적: K-Note를 게시판에 게시 가능한 마크다운 문서로 변환
모델: Azure OpenAI GPT-4.1-mini
```
- **입력**: K-Note JSON
- **LLM 프롬프트**: `KNOTE_TO_STANDARD_DOC_PROMPT`
  - K-Note 필드 → 표준 문서 섹션 매핑
  - 메타데이터 추출 (제목, 작성자, 버전, 태그)
  - 7개 섹션 본문 생성 (목적, 배경, 접근방법, 결과, 한계, 적용방안, 참고자료)
- **처리**:
  - K-Note → 마크다운 문서 변환
  - Azure Blob Storage `enhanced` 컨테이너에 업로드
  - 벡터화 (SemanticChunker) → ChromaDB 저장
  - 게시판 DB (SQLite)에 메타데이터 저장
- **출력**:
  - 마크다운 문서 (Azure Blob Storage)
  - 벡터 임베딩 (ChromaDB)
  - 게시글 (board.db)

---

#### **워크플로우 요약**

```
VectorDB 품질 필터링 → MMR 다양성 샘플링 → LLM 구조화 요약
    ↓
RAG 컨텍스트 확장 → GPT-4o 아날로지 융합 → 반례 검증
    ↓
K-Note 표준화 → 품질 평가 (반복 조건) → 마크다운 구체화
    ↓
Azure Blob Storage + ChromaDB + 게시판 저장
```

**사용 Azure 서비스:**
- **Azure OpenAI (GPT-4o)**: 지식 융합 (Synthesizer Agent)
- **Azure OpenAI (GPT-4.1-mini)**: 요약, 검증, K-Note 생성, 문서 구체화
- **Azure OpenAI Embeddings**: MMR 샘플링, RAG 검색
- **ChromaDB**: 벡터 검색 및 유사도 계산
- **Azure Blob Storage**: 구체화된 K-Note 문서 저장

#### Multi-Agent 역할

| Agent | Model | Temperature | 역할 |
|-------|-------|-------------|------|
| **Librarian** | GPT-4.1-mini | 0.0 | 데이터 정규화 및 품질 필터링 |
| **Sampler** | - | - | MMR 기반 다양성 샘플링 |
| **Summarizer** | GPT-4.1-mini | 0.0 | 구조화 요약 생성 |
| **Expander** | - | - | RAG 기반 컨텍스트 확장 |
| **Synthesizer** | GPT-4o | 0.3 (설정 가능) | 아날로지 기반 지식 융합 |
| **Verifier** | GPT-4.1-mini | 0.0 | 제안 검증 및 반례 검토 |
| **Productizer** | GPT-4.1-mini | 0.1 | K-Note 표준 형식 변환 |
| **Evaluator** | - | - | 품질 점수 평가 |

## 📁 프로젝트 구조

```
ktds-610-msai-mvp/
├── app.py                          # 메인 Streamlit 애플리케이션
│
├── knowledge/                      # 📚 지식 등록 모듈
│   ├── __init__.py
│   ├── service.py                  # 문서 분석/보완/벡터화 서비스
│   └── prompts.py                  # LLM 프롬프트 템플릿
│
├── knowledge_creation/             # 🔬 지식 창출 모듈
│   ├── __init__.py
│   └── creation_engine.py          # LangGraph Multi-Agent 워크플로우 엔진
│
├── board/                          # 📋 게시판 모듈
│   ├── __init__.py
│   └── service.py                  # 게시글 조회/필터링/통계 서비스
│
├── config/                         # ⚙️ 설정 모듈
│   ├── __init__.py
│   └── settings.py                 # LLM, VectorDB, Document 설정
│
├── utils/                          # 🛠️ 유틸리티 모듈
│   ├── __init__.py
│   ├── file_processor.py           # Azure Blob Storage 업로드
│   └── session_manager.py          # Streamlit 세션 상태 관리
│
├── data/                           # 💾 데이터 저장소
│   ├── chroma_db/                  # ChromaDB 영속성 디렉토리
│   └── board.db                    # SQLite 게시판 데이터베이스
│
├── .env.example                    # 환경변수 템플릿
├── .env                            # 환경변수 (gitignore)
├── init_db.py                      # SQLite DB 초기화 스크립트
└── requirements.txt                # Python 의존성 목록
```

### 디렉토리별 상세 설명

#### 📚 `knowledge/` - 지식 등록
- **역할**: 문서 업로드 → AI 분석 → 보완 → 벡터화 → 저장
- **주요 기능**:
  - `service.py`: MarkItDown/Azure DI 문서 변환, Azure OpenAI 분석/보완, SemanticChunker 청킹, ChromaDB 저장
  - `prompts.py`: 문서 분석 프롬프트, 보완 문서 생성 프롬프트
- **의존성**: Azure OpenAI, Azure Document Intelligence, Azure Blob Storage, ChromaDB

#### 🔬 `knowledge_creation/` - 지식 창출
- **역할**: Multi-Agent 시스템으로 기존 지식에서 새로운 지식(K-Note) 창출
- **주요 기능**:
  - `creation_engine.py`: LangGraph 8단계 워크플로우 (정규화 → 샘플링 → 요약 → RAG 확장 → 융합 → 검증 → K-Note 생성 → 평가)
  - 8개 Agent: Librarian, Sampler, Summarizer, Expander, Synthesizer, Verifier, Productizer, Evaluator
  - MMR 샘플링, RAG 컨텍스트 확장, 아날로지 기반 지식 융합
- **의존성**: LangGraph, Azure OpenAI (GPT-4o, GPT-4.1-mini), ChromaDB

#### 📋 `board/` - 게시판
- **역할**: 보완된 지식 문서 조회 및 관리
- **주요 기능**:
  - `service.py`: SQLite CRUD, 필터링 (품질/날짜/작성자/검색어), 정렬, 통계, 내보내기 (TXT/CSV/JSON)
  - 조회수 추적, 품질 점수 재계산, Azure Blob Storage 다운로드 링크 관리
- **의존성**: SQLite, Azure Blob Storage (다운로드 URL)

#### ⚙️ `config/` - 설정
- **역할**: 애플리케이션 전역 설정 관리
- **주요 설정**:
  - `settings.py`: LLM 모델 설정 (Azure OpenAI), VectorDB 설정 (dimension, threshold), Document 설정 (파일 크기, 지원 형식, 청크 설정)
- **환경변수**: .env 파일에서 API 키, 엔드포인트 로드

#### 🛠️ `utils/` - 유틸리티
- **역할**: 공통 기능 제공
- **주요 기능**:
  - `file_processor.py`: 파일 유효성 검사, Azure Blob Storage 업로드 (original/enhanced 컨테이너)
  - `session_manager.py`: Streamlit 세션 상태 초기화 및 관리
- **의존성**: Azure Blob Storage, Streamlit

#### 💾 `data/` - 데이터 저장소
- **`chroma_db/`**: ChromaDB 영속성 디렉토리 (벡터 인덱스, 메타데이터)
- **`board.db`**: SQLite 데이터베이스 (게시글 테이블: id, title, content, author, timestamp, views, quality_score, original_url, enhanced_url)

## ⚙️ 설치 및 실행

### 필수 요구사항
- Python 3.10+
- Azure AI Services 계정
  - Azure OpenAI Service
  - Azure Document Intelligence
  - Azure Blob Storage

### 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
# .env 파일 생성 후 아래 변수 추가
```

### 환경 변수 (.env)

```bash
# Azure OpenAI
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_AI_FOUNDRY_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_GPT4O_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=your-connection-string
AZURE_STORAGE_ORIGINAL_CONTAINER_NAME=original
AZURE_STORAGE_ENHANCED_CONTAINER_NAME=enhanced
```

### 실행

```bash
# 애플리케이션 실행
streamlit run app.py

# 특정 포트로 실행
streamlit run app.py --server.port 8501
```

### 데이터베이스 초기화

```bash
# SQLite 게시판 DB 초기화 (선택사항)
python init_db.py
```

## 🔄 워크플로우 요약

```
[지식 등록]
문서 업로드 → AI 분석/보완 → Azure Blob Storage 저장 → 벡터화 → ChromaDB/게시판 저장

[게시판]
SQLite 조회 → 게시글 표시 → Azure Blob Storage에서 다운로드

[지식 창출]
ChromaDB 샘플링 → Multi-Agent 워크플로우(8단계) → K-Note 생성 → 문서 구체화 → 저장
```

## 🛠️ 주요 라이브러리

- **streamlit**: 웹 UI 프레임워크
- **langchain**: LLM 애플리케이션 프레임워크
- **langgraph**: Multi-Agent 워크플로우 오케스트레이션
- **chromadb**: 벡터 데이터베이스
- **openai**: Azure OpenAI API
- **markitdown**: 범용 문서 → 마크다운 변환
- **azure-ai-documentintelligence**: PDF/Office 파일 처리
- **azure-storage-blob**: Azure Blob Storage 연동

## 📊 성능 특징

- **MMR 샘플링**: 품질·다양성·최신성 균형 잡힌 샘플 선택
- **SemanticChunker**: 의미 기반 청크 분할로 컨텍스트 보존
- **Multi-Agent 반복**: 품질 임계값 미달 시 최대 3회 반복 실행
- **ChromaDB 영속성**: 벡터 데이터 영구 저장 및 재사용

## 📝 라이센스

교육 목적으로 개발된 프로젝트입니다.
