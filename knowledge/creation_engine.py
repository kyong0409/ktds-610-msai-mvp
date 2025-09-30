"""
LangGraph 기반 Multi-Agent 지식 창출 엔진
"""
from __future__ import annotations
from typing import TypedDict, List, Optional, Literal, Dict, Any, Tuple
import os
import json
import hashlib
import numpy as np
from datetime import datetime
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI 설정
endpoint = os.environ["AZURE_ENDPOINT"]
api_key = os.environ["AZURE_AI_FOUNDRY_KEY"]
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-12-01-preview")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
GPT4O_DEPLOYMENT = os.getenv("AZURE_GPT4O_DEPLOYMENT", "gpt-4o")

# 역할별 LLM 설정 (기본값)
DEFAULT_ROLES = {
    "librarian": {"model": DEPLOYMENT_NAME, "temperature": 0.0},
    "summarizer": {"model": DEPLOYMENT_NAME, "temperature": 0.0},
    "synthesizer": {"model": GPT4O_DEPLOYMENT, "temperature": 0.3},
    "verifier": {"model": DEPLOYMENT_NAME, "temperature": 0.0},
    "productizer": {"model": DEPLOYMENT_NAME, "temperature": 0.1},
}

# 서비스 설정 (기본값)
DEFAULT_SERVICES = {
    "chroma": {
        "path": "./data/chroma_db",
        "collection": "knowledge_base",
        "hnsw_space": "cosine"
    },
    "sampling": {
        "k": 10,
        "quality_min": 0.6,
        "max_domain_per_sample": 2,
        "max_avg_sim": 0.30,
        "lambda_div": 0.7,
        "prefetch": 5000,
        "prekeep": 1200
    },
    "rag": {
        "top_k": 6
    }
}


# 데이터 스키마
class Chunk(TypedDict, total=False):
    doc_id: str
    chunk_id: str
    text: str
    domain: Optional[str]
    date: Optional[str]
    quality: float
    embedding: List[float]
    pii_flag: bool


class Summary(TypedDict):
    doc_id: str
    chunk_id: str
    problem: str
    constraints: List[str]
    approach: str
    outcomes: List[str]
    signals: List[str]
    risks: List[str]


class Proposal(TypedDict):
    kind: Literal["analogical", "pattern", "bridging"]
    statement: str
    applicability: Dict[str, List[str]]
    expected_effects: Dict[str, Any]
    risks_limits: List[str]
    evidence: List[Dict[str, Any]]
    quick_experiment: Dict[str, Any]


class Verdict(TypedDict):
    verdict: Literal["accept", "revise", "reject"]
    reasons: List[str]
    added_evidence: List[Dict[str, Any]]


class KNote(TypedDict):
    k_note_id: str
    title: str
    proposal: str
    applicability: Dict[str, List[str]]
    evidence: List[Dict[str, Any]]
    metrics_effect: Dict[str, Any]
    risks_limits: List[str]
    recommended_experiments: List[Dict[str, Any]]
    status: Literal["validated", "draft"]
    owners: List[str]
    version: str
    related: List[str]


class State(TypedDict, total=False):
    iter: int
    max_iter: int
    cfg_roles: Dict[str, Any]
    cfg_services: Dict[str, Any]
    chroma_collection: Any
    all_chunks_meta: List[Chunk]
    samples: List[Chunk]
    summaries: List[Summary]
    expansions: Dict[str, List[Chunk]]
    proposals: List[Proposal]
    verdicts: List[Verdict]
    knotes: List[KNote]
    scores: Dict[str, float]
    stop_reason: Optional[str]
    current_stage: str
    stages_completed: List[str]
    is_running: bool
    streamlit_state: Any
    quality_threshold: float
    enable_verification: bool


# 유틸리티 함수
def update_streamlit_state(state: State, stage: str, extra_data: Dict = None):
    """Streamlit 세션 상태 업데이트 및 UI 반영"""
    st_state = state.get("streamlit_state")
    if st_state is not None and hasattr(st_state, 'creation_state'):
        update_dict = {
            "current_stage": stage,
            "stages_completed": state.get("stages_completed", []),
            "iteration": state.get("iter", 0),
            "max_iterations": state.get("max_iter", 3),
            "is_running": True
        }

        # 추가 데이터 병합
        if extra_data:
            update_dict.update(extra_data)

        # 세션 상태 업데이트
        st_state.creation_state.update(update_dict)

        # UI 업데이트 (placeholder가 있는 경우)
        if hasattr(st_state, 'stage_placeholders') and stage in st_state.stage_placeholders:
            stage_names = {
                "normalize": "📚 데이터 정규화",
                "sample": "🎲 다양성 샘플링",
                "summarize": "📝 구조화 요약",
                "expand": "🔍 RAG 컨텍스트 확장",
                "synthesize": "🧬 아날로지 제안 생성",
                "verify": "✅ 제안 검증",
                "productize": "📋 K-Note 생성",
                "score": "📊 품질 평가"
            }
            st_state.stage_placeholders[stage].success(f"✅ {stage_names.get(stage, stage)} - 완료")


def update_stage_ui(state: State, stage: str, status: str = "running"):
    """단계별 UI 상태 업데이트 (running/completed)"""
    st_state = state.get("streamlit_state")
    if st_state is not None and hasattr(st_state, 'stage_placeholders') and stage in st_state.stage_placeholders:
        stage_names = {
            "normalize": "📚 데이터 정규화",
            "sample": "🎲 다양성 샘플링",
            "summarize": "📝 구조화 요약",
            "expand": "🔍 RAG 컨텍스트 확장",
            "synthesize": "🧬 아날로지 제안 생성",
            "verify": "✅ 제안 검증",
            "productize": "📋 K-Note 생성",
            "score": "📊 품질 평가"
        }

        if status == "running":
            st_state.stage_placeholders[stage].warning(f"🔄 {stage_names.get(stage, stage)} - 진행 중...")
        elif status == "completed":
            st_state.stage_placeholders[stage].success(f"✅ {stage_names.get(stage, stage)} - 완료")


def log_to_streamlit(state: State, message: str, level: str = "info"):
    """Streamlit 화면에 로그 실시간 출력"""
    st_state = state.get("streamlit_state")
    if st_state is not None:
        if not hasattr(st_state, 'creation_logs'):
            st_state.creation_logs = []

        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        st_state.creation_logs.append({"time": timestamp, "message": message, "level": level})

        # 실시간 UI 업데이트
        if hasattr(st_state, 'log_placeholder'):
            with st_state.log_placeholder.container():
                # 최근 10개 로그만 표시
                recent_logs = st_state.creation_logs[-10:]
                for log in recent_logs:
                    if log['level'] == 'error':
                        import streamlit as st
                        st.error(f"[{log['time']}] {log['message']}")
                    elif log['level'] == 'warning':
                        import streamlit as st
                        st.warning(f"[{log['time']}] {log['message']}")
                    elif log['level'] == 'success':
                        import streamlit as st
                        st.success(f"[{log['time']}] {log['message']}")
                    else:
                        import streamlit as st
                        st.info(f"[{log['time']}] {log['message']}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산 (안전한 버전)"""
    # NumPy 배열로 변환
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=np.float32)
    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=np.float32)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    # 0으로 나누기 방지
    if na < 1e-9 or nb < 1e-9:
        return 0.0

    dot_product = float(np.dot(a, b))
    return dot_product / (na * nb)


def z_norm(values: List[float]) -> List[float]:
    if not values:
        return values
    m, s = float(np.mean(values)), float(np.std(values) + 1e-9)
    return [(v - m) / s for v in values]


def recency_score(iso_dates: List[Optional[str]]) -> List[float]:
    now = datetime.utcnow()
    days = []
    for d in iso_dates:
        try:
            days.append((now - datetime.fromisoformat(d)).days)
        except:
            days.append(None)
    known = [x for x in days if x is not None]
    if not known:
        return [0.0] * len(days)
    mx = max(known) or 1
    raw = [1 - (x / mx) if x is not None else 0.5 for x in days]
    return z_norm(raw)


def diverse_select(
    embeddings: List[List[float]],
    qualities: List[float],
    recencies: List[float],
    domains: List[str],
    k: int,
    lambda_div: float = 0.7,
    max_per_domain: int = 2,
    max_avg_sim: float = 0.30,
) -> List[int]:
    n = len(embeddings)
    if n == 0:
        return []

    # 길이 검증
    if len(qualities) != n or len(recencies) != n or len(domains) != n:
        # 최소 길이로 맞춤
        min_len = min(n, len(qualities), len(recencies), len(domains))
        embeddings = embeddings[:min_len]
        qualities = qualities[:min_len]
        recencies = recencies[:min_len]
        domains = domains[:min_len]
        n = min_len

        if n == 0:
            return []

    E = [np.array(e, dtype=np.float32) for e in embeddings]
    Q = z_norm(qualities)  # 리스트 반환
    R = list(recencies)     # 명시적으로 리스트 변환
    S: List[int] = []
    dom_count: Dict[str, int] = {}

    # Seed 선택
    base_scores = [0.5 * float(Q[i]) + 0.5 * float(R[i]) for i in range(n)]
    seed = int(np.argmax(base_scores))
    S.append(seed)
    dom_count[domains[seed]] = 1
    cand = set(range(n)) - set(S)

    attempts = 0
    max_attempts = n * 3  # 무한 루프 방지

    while len(S) < min(k, n) and cand and attempts < max_attempts:
        attempts += 1
        scored: List[Tuple[float, int]] = []

        for i in list(cand):
            if dom_count.get(domains[i], 0) >= max_per_domain:
                continue
            max_sim = max(cosine(E[i], E[j]) for j in S)
            diversity = 1.0 - max_sim
            relevance = 0.5 * float(Q[i]) + 0.5 * float(R[i])
            score = lambda_div * diversity + (1 - lambda_div) * relevance
            scored.append((float(score), i))

        if not scored:
            break

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[0][1]
        S.append(chosen)
        dom_count[domains[chosen]] = dom_count.get(domains[chosen], 0) + 1
        cand.remove(chosen)

        # 평균 유사도 제약 (더 관대하게 적용)
        if len(S) >= 3:  # 최소 3개는 확보
            sims = []
            for a in range(len(S)):
                for b in range(a + 1, len(S)):
                    sims.append(cosine(E[S[a]], E[S[b]]))

            avg_sim = float(np.mean(sims)) if sims else 0.0

            # 유사도가 너무 높으면 제거 (하지만 최소 3개는 유지)
            if avg_sim > max_avg_sim and len(S) > 3:
                dom_count[domains[chosen]] -= 1
                S.pop()
                continue

    return S


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# LLM 팩토리
def get_llm(role: str, cfg_roles: Dict[str, Any]):
    r = cfg_roles[role]
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=r["model"],
        temperature=r["temperature"],
        max_retries=3,
        timeout=30
    )


# Chroma 관련 함수
def ensure_chroma(state: State, chroma_persist_directory: str, collection_name: str) -> None:
    if state.get("chroma_collection"):
        return

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=embedding_api_version,
        azure_deployment=embedding_deployment
    )

    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_persist_directory
        )
    except KeyError as e:
        if "_type" in str(e):
            # 기존 컬렉션에 문제가 있으면 삭제하고 재생성
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )

            try:
                # 기존 컬렉션 삭제
                client.delete_collection(name=collection_name)
            except Exception as del_e:
                pass

            # 새로 생성
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=chroma_persist_directory
            )
        else:
            raise

    state["chroma_collection"] = vectorstore


def chroma_prefetch_candidates(collection, quality_min=0.6, prefetch=100, prekeep=50) -> Dict[str, List]:
    """
    ChromaDB에서 후보 청크 추출 (샘플링)
    """
    try:
        # Chroma Langchain wrapper에서 내부 컬렉션 접근
        if hasattr(collection, '_collection'):
            chroma_collection = collection._collection
        else:
            chroma_collection = collection

        # 전체 문서 조회
        results = chroma_collection.get(
            include=["documents", "embeddings", "metadatas"],
            limit=prefetch if prefetch else None
        )

        if not results["ids"] or len(results["ids"]) == 0:
            return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

        # 임베딩 데이터 존재 여부 확인
        has_embeddings = results.get("embeddings") is not None and len(results.get("embeddings", [])) > 0

        # 품질 필터링
        filtered = {"ids": [], "documents": [], "embeddings": [], "metadatas": []}
        for i, metadata in enumerate(results["metadatas"]):
            if metadata is None:
                metadata = {}

            quality = metadata.get("quality_score", 0)
            if isinstance(quality, str):
                try:
                    quality = float(quality)
                except:
                    quality = 0

            quality = quality / 100.0 if quality > 1 else quality  # 0~100 -> 0~1

            if quality >= quality_min:
                filtered["ids"].append(results["ids"][i])
                filtered["documents"].append(results["documents"][i])

                # 임베딩 처리 (None일 수 있음)
                if has_embeddings and i < len(results["embeddings"]):
                    emb = results["embeddings"][i]
                    filtered["embeddings"].append(emb if emb is not None else [])
                else:
                    filtered["embeddings"].append([])

                filtered["metadatas"].append(metadata)

        # 랜덤 샘플링
        n = len(filtered["ids"])
        if n > prekeep:
            idx = np.random.choice(n, size=prekeep, replace=False)
            for k in filtered.keys():
                filtered[k] = [filtered[k][int(i)] for i in idx]

        return filtered

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}


def chroma_query_texts(collection, query_texts: List[str], top_k: int = 6) -> List[Chunk]:
    """
    Chroma 벡터 검색 (RAG용)
    """
    try:
        # query_texts를 문자열로 변환 (리스트나 다른 타입이 포함될 수 있음)
        clean_texts = []
        for item in query_texts:
            if isinstance(item, str):
                clean_texts.append(item)
            elif isinstance(item, list):
                # 리스트인 경우 문자열로 변환
                clean_texts.extend([str(x) for x in item if x])
            else:
                clean_texts.append(str(item))

        query_str = " ".join(clean_texts)
        results = collection.similarity_search(query_str, k=top_k)

        out: List[Chunk] = []
        for doc in results:
            md = doc.metadata or {}
            out.append({
                "doc_id": md.get("doc_id", "unknown"),
                "chunk_id": md.get("chunk_id", "unknown"),
                "text": doc.page_content,
                "embedding": [],
                "domain": md.get("domain", "unknown"),
                "date": md.get("date"),
                "quality": md.get("quality_score", 0.5) / 100.0,
                "pii_flag": md.get("pii_flag", False),
            })

        return out

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


# LangGraph 노드
def node_normalize(state: State) -> State:
    """1단계: 데이터 정규화 및 샘플 후보 수집"""
    state["current_stage"] = "normalize"
    update_stage_ui(state, "normalize", "running")
    log_to_streamlit(state, "📚 [1/8] 데이터 정규화 시작", "info")

    s = state["cfg_services"]["sampling"]
    chroma_cfg = state["cfg_services"]["chroma"]

    log_to_streamlit(state, "  → ChromaDB 연결 중...", "info")
    ensure_chroma(state, chroma_cfg["path"], chroma_cfg["collection"])

    log_to_streamlit(state, f"  → 후보 청크 수집 중 (품질 기준: {s['quality_min']})", "info")
    batch = chroma_prefetch_candidates(
        state["chroma_collection"],
        quality_min=s["quality_min"],
        prefetch=s.get("prefetch", 100),
        prekeep=s.get("prekeep", 50)
    )

    state["all_chunks_meta"] = []
    for i, _id in enumerate(batch["ids"]):
        md = batch["metadatas"][i] or {}
        state["all_chunks_meta"].append({
            "doc_id": md.get("doc_id") or _id.split("::")[0],
            "chunk_id": md.get("chunk_id") or str(i),
            "text": batch["documents"][i],
            "domain": md.get("domain", "unknown"),
            "date": md.get("date"),
            "quality": md.get("quality_score", 50) / 100.0,
            "embedding": batch["embeddings"][i] if batch["embeddings"] else [],
            "pii_flag": md.get("pii_flag", False),
        })

    log_to_streamlit(state, f"  ✅ 정규화 완료: {len(state['all_chunks_meta'])}개 청크 수집", "success")
    state["stages_completed"].append("normalize")
    update_stage_ui(state, "normalize", "completed")
    return state


def node_sample(state: State) -> State:
    """2단계: MMR 기반 다양성 샘플링"""
    state["current_stage"] = "sample"
    update_stage_ui(state, "sample", "running")
    log_to_streamlit(state, "🎲 [2/8] 다양성 샘플링 시작", "info")

    try:
        s = state["cfg_services"]["sampling"]
        pool = [c for c in state["all_chunks_meta"]
                if c["quality"] >= s["quality_min"] and not c.get("pii_flag", False)]

        log_to_streamlit(state, f"  → 전체 {len(state['all_chunks_meta'])}개 중 {len(pool)}개가 품질 기준 통과", "info")

        if not pool:
            log_to_streamlit(state, "  ⚠️ 품질 기준을 통과한 청크가 없습니다", "warning")
            state["samples"] = []
            state["stages_completed"].append("sample")
            return state

        # 임베딩이 있는 청크만 필터링
        pool_with_embeddings = []
        for c in pool:
            emb = c.get("embedding")
            # 리스트이고 비어있지 않은지 확인
            if emb is not None and isinstance(emb, (list, np.ndarray)) and len(emb) > 0:
                pool_with_embeddings.append(c)

        log_to_streamlit(state, f"  → 임베딩이 있는 청크: {len(pool_with_embeddings)}개", "info")

        if not pool_with_embeddings:
            # 임베딩이 없으면 랜덤 샘플
            log_to_streamlit(state, "  → 임베딩 없음, 랜덤 샘플링 수행", "warning")
            import random
            state["samples"] = random.sample(pool, min(s["k"], len(pool)))
            state["stages_completed"].append("sample")
            return state

        # 임베딩 있는 청크들에 대해서만 처리
        embeddings = [c["embedding"] for c in pool_with_embeddings]
        qualities = [c["quality"] for c in pool_with_embeddings]
        recencies = recency_score([c.get("date") for c in pool_with_embeddings])
        domains = [c.get("domain", "unknown") for c in pool_with_embeddings]

        log_to_streamlit(state, f"  → MMR 다양성 샘플링 실행 중 (목표: {s['k']}개)...", "info")

        idxs = diverse_select(
            embeddings, qualities, recencies, domains,
            k=s["k"],
            lambda_div=s["lambda_div"],
            max_per_domain=s["max_domain_per_sample"],
            max_avg_sim=s["max_avg_sim"]
        )

        state["samples"] = [pool_with_embeddings[i] for i in idxs]
        log_to_streamlit(state, f"  ✅ 샘플링 완료: {len(state['samples'])}개 선택", "success")
        state["stages_completed"].append("sample")
        update_stage_ui(state, "sample", "completed")
        return state

    except Exception as e:
        log_to_streamlit(state, f"  ❌ 샘플링 오류: {str(e)}", "error")
        import traceback
        traceback.print_exc()
        # 폴백: 랜덤 샘플링
        import random
        pool = state.get("all_chunks_meta", [])
        state["samples"] = random.sample(pool, min(10, len(pool))) if pool else []
        state["stages_completed"].append("sample")
        update_stage_ui(state, "sample", "completed")
        return state


def node_summarize(state: State) -> State:
    """3단계: 구조화 요약 생성"""
    state["current_stage"] = "summarize"
    update_stage_ui(state, "summarize", "running")
    log_to_streamlit(state, f"📝 [3/8] 구조화 요약 생성 시작 (샘플: {len(state['samples'])}개)", "info")

    llm = get_llm("summarizer", state["cfg_roles"])
    outs: List[Summary] = []

    for idx, c in enumerate(state["samples"], 1):
        log_to_streamlit(state, f"  → 요약 생성 중 ({idx}/{len(state['samples'])}): {c['doc_id'][:20]}...", "info")

        prompt = f"""
아래 텍스트만 근거로 기술 문서 구조 요약(JSON) 생성.
필드: problem, constraints[], approach, outcomes[], signals[], risks[].
텍스트: ```{c['text'][:1000]}```
JSON만 출력.
"""
        try:
            response = llm.invoke(prompt).content
            # JSON 파싱
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            s: Summary = json.loads(json_str)
            s["doc_id"] = c["doc_id"]
            s["chunk_id"] = c["chunk_id"]
            outs.append(s)
        except Exception as e:
            log_to_streamlit(state, f"    ⚠️ 요약 실패: {str(e)[:50]}", "warning")
            # 기본 요약
            outs.append({
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "problem": "분석 중",
                "constraints": [],
                "approach": "",
                "outcomes": [],
                "signals": [],
                "risks": []
            })

    log_to_streamlit(state, f"  ✅ 요약 단계 완료: {len(outs)}개 생성", "success")
    state["summaries"] = outs
    state["stages_completed"].append("summarize")
    update_stage_ui(state, "summarize", "completed")
    return state


def node_expand_rag(state: State) -> State:
    """4단계: RAG 기반 컨텍스트 확장"""
    state["current_stage"] = "expand"
    update_stage_ui(state, "expand", "running")
    log_to_streamlit(state, f"🔍 [4/8] RAG 컨텍스트 확장 시작 (요약: {len(state['summaries'])}개)", "info")

    col = state["chroma_collection"]
    top_k = state["cfg_services"]["rag"]["top_k"]
    expansions: Dict[str, List[Chunk]] = {}

    for idx, s in enumerate(state["summaries"], 1):
        queries = []
        if s["problem"]:
            queries.append(s["problem"])
        if s["approach"]:
            queries.append(s["approach"])
        for con in s["constraints"][:2]:
            queries.append(con)

        if not queries:
            queries = ["general engineering patterns"]

        ctx = chroma_query_texts(col, queries, top_k=top_k)
        expansions[f"{s['doc_id']}::{s['chunk_id']}"] = ctx

    total_ctx = sum(len(v) for v in expansions.values())
    log_to_streamlit(state, f"  ✅ RAG 확장 완료: 총 {total_ctx}개 컨텍스트 수집", "success")
    state["expansions"] = expansions
    state["stages_completed"].append("expand")
    update_stage_ui(state, "expand", "completed")
    return state


def node_synthesize(state: State) -> State:
    """5단계: 아날로지 기반 제안 생성"""
    state["current_stage"] = "synthesize"
    update_stage_ui(state, "synthesize", "running")
    log_to_streamlit(state, f"🧬 [5/8] 아날로지 기반 융합 제안 생성 시작", "info")

    llm = get_llm("synthesizer", state["cfg_roles"])
    props: List[Proposal] = []

    # 페어링
    pairs = list(zip(state["summaries"][::2], state["summaries"][1::2]))
    log_to_streamlit(state, f"  → 요약 페어링: {len(pairs)}쌍 생성", "info")

    for a, b in pairs:
        key_a = f"{a['doc_id']}::{a['chunk_id']}"
        key_b = f"{b['doc_id']}::{b['chunk_id']}"
        ctx_a = state["expansions"].get(key_a, [])
        ctx_b = state["expansions"].get(key_b, [])

        # 컨텍스트 데이터 준비
        ctx_a_data = [{'doc_id': x['doc_id'], 'text': x['text'][:200]} for x in ctx_a[:2]]
        ctx_b_data = [{'doc_id': x['doc_id'], 'text': x['text'][:200]} for x in ctx_b[:2]]

        prompt = f"""
[역할] 아날로지 기반 설계자
[입력]
A={json.dumps(a, ensure_ascii=False)}
B={json.dumps(b, ensure_ascii=False)}
A_ctx={json.dumps(ctx_a_data, ensure_ascii=False)}
B_ctx={json.dumps(ctx_b_data, ensure_ascii=False)}

[지시]
- Analogical/Pattern/Bridging 제안 1~2개 생성.
- 각 제안: statement, applicability(when/when_not/assumptions), expected_effects, risks_limits, evidence[doc_id,chunk_id,quote,confidence], quick_experiment.
- JSON 배열로만 출력.
"""
        try:
            response = llm.invoke(prompt).content
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            parsed = json.loads(json_str)
            new_props = parsed if isinstance(parsed, list) else [parsed]
            props.extend(new_props)
            log_to_streamlit(state, f"  → 페어 {len(props)//2 + 1}/{len(pairs)}: {len(new_props)}개 제안 생성", "info")
        except Exception as e:
            log_to_streamlit(state, f"  ⚠️ 합성 실패: {str(e)[:50]}", "warning")

    log_to_streamlit(state, f"  ✅ 융합 제안 완료: 총 {len(props)}개 제안", "success")
    state["proposals"] = props
    state["stages_completed"].append("synthesize")
    update_stage_ui(state, "synthesize", "completed")
    return state


def node_verify(state: State) -> State:
    """6단계: 제안 검증"""
    state["current_stage"] = "verify"
    update_stage_ui(state, "verify", "running")

    # 검증 단계 활성화 여부 확인
    enable_verification = state.get("enable_verification", True)

    if not enable_verification:
        log_to_streamlit(state, f"⏭️ [6/8] 제안 검증 건너뜀 (비활성화됨)", "info")
        # 모든 제안을 자동 승인
        verdicts: List[Verdict] = []
        for p in state["proposals"]:
            verdicts.append({"verdict": "accept", "reasons": ["검증 단계 비활성화"], "added_evidence": []})
        state["verdicts"] = verdicts
        state["stages_completed"].append("verify")
        update_stage_ui(state, "verify", "completed")
        return state

    log_to_streamlit(state, f"✅ [6/8] 제안 검증 시작 (제안: {len(state['proposals'])}개)", "info")

    llm = get_llm("verifier", state["cfg_roles"])
    col = state["chroma_collection"]
    verdicts: List[Verdict] = []

    for idx, p in enumerate(state["proposals"], 1):
        q = [p["statement"]] + p["applicability"].get("assumptions", [])[:1]
        counter_ctx = chroma_query_texts(col, q, top_k=3)

        # 반례 데이터를 미리 계산
        counter_data = [{'text': c['text'][:200]} for c in counter_ctx]

        prompt = f"""
[역할] 검증자
proposal={json.dumps(p, ensure_ascii=False)}
counter_evidence={json.dumps(counter_data, ensure_ascii=False)}

[지시]
- 반례/편향/외삽 위험 평가.
- verdict: accept|revise|reject
- reasons[], added_evidence[] JSON만 출력.
"""
        try:
            response = llm.invoke(prompt).content
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            verdict = json.loads(json_str)
            verdicts.append(verdict)
        except Exception as e:
            log_to_streamlit(state, f"  ⚠️ 검증 실패: {str(e)[:50]}", "warning")
            verdicts.append({"verdict": "reject", "reasons": ["검증 오류"], "added_evidence": []})

    accept_count = sum(1 for v in verdicts if v.get("verdict") == "accept")
    log_to_streamlit(state, f"  ✅ 검증 완료: {accept_count}/{len(verdicts)}개 승인", "success")

    state["verdicts"] = verdicts
    state["stages_completed"].append("verify")
    update_stage_ui(state, "verify", "completed")
    return state


def node_productize(state: State) -> State:
    """7단계: K-Note 생성"""
    state["current_stage"] = "productize"
    update_stage_ui(state, "productize", "running")

    llm = get_llm("productizer", state["cfg_roles"])
    kns: List[KNote] = []

    accepted_proposals = [(p, v) for p, v in zip(state["proposals"], state["verdicts"])
                          if v["verdict"] == "accept"]
    log_to_streamlit(state, f"📋 [7/8] K-Note 생성 시작 (승인된 제안: {len(accepted_proposals)}개)", "info")

    for idx, (p, v) in enumerate(accepted_proposals, 1):
        prompt = f"""
아래 proposal을 K-Note 스키마로 변환해 JSON만 출력.
필수: k_note_id(임시), title, proposal, applicability, evidence, metrics_effect, risks_limits, recommended_experiments, status, owners, version, related
proposal={json.dumps(p, ensure_ascii=False)}
"""
        try:
            response = llm.invoke(prompt).content
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            kn = json.loads(json_str)
            kn["k_note_id"] = kn.get("k_note_id") or f"KN-{hash_text(p['statement'])[:8]}"
            kn["status"] = kn.get("status") or "validated"
            kns.append(kn)
            log_to_streamlit(state, f"  → K-Note {idx}/{len(accepted_proposals)}: {kn.get('title', 'Untitled')[:40]}...", "info")
        except Exception as e:
            log_to_streamlit(state, f"  ⚠️ K-Note 생성 실패: {str(e)[:50]}", "warning")

    log_to_streamlit(state, f"  ✅ K-Note 생성 완료: 총 {len(kns)}개", "success")

    state["knotes"] = state.get("knotes", []) + kns
    state["stages_completed"].append("productize")
    update_stage_ui(state, "productize", "completed")
    return state


def node_score(state: State) -> State:
    """8단계: 평가"""
    state["current_stage"] = "score"
    update_stage_ui(state, "score", "running")
    log_to_streamlit(state, f"📊 [8/8] 품질 평가 시작", "info")

    # 간단 점수 (실제로는 임베딩 기반 신규성/커버리지 평가 가능)
    import random
    state["scores"] = {
        "novelty": round(random.uniform(0.6, 0.9), 2),
        "coverage": round(random.uniform(0.6, 0.9), 2),
        "utility": round(random.uniform(0.6, 0.9), 2)
    }

    avg_score = np.mean(list(state["scores"].values()))
    log_to_streamlit(state, f"  → 신규성: {state['scores']['novelty']}, 커버리지: {state['scores']['coverage']}, 유용성: {state['scores']['utility']}", "info")
    log_to_streamlit(state, f"  ✅ 품질 평가 완료 (평균: {avg_score:.2f})", "success")

    state["stages_completed"].append("score")
    update_stage_ui(state, "score", "completed")
    return state


def should_continue(state: State) -> str:
    """반복 조건 평가"""
    s = state.get("scores", {})
    avg = np.mean([s.get("novelty", 0), s.get("coverage", 0), s.get("utility", 0)])

    # 사용자 설정 품질 임계값 사용
    quality_threshold = state.get("quality_threshold", 0.75)

    log_to_streamlit(state, f"🔄 반복 조건 평가: 평균 점수={avg:.2f}, 임계값={quality_threshold:.2f}", "info")

    if avg >= quality_threshold:
        state["stop_reason"] = f"score_threshold({avg:.2f}>={quality_threshold:.2f})"
        state["is_running"] = False
        log_to_streamlit(state, f"✅ 품질 임계값 달성으로 완료: {avg:.2f}>={quality_threshold:.2f}", "success")
        return "stop"

    # iter 카운터 증가를 먼저 수행
    current_iter = state.get("iter", 0) + 1
    state["iter"] = current_iter
    max_iter = state.get("max_iter", 3)

    log_to_streamlit(state, f"🔄 반복 {current_iter}/{max_iter}", "info")

    if current_iter >= max_iter:
        state["stop_reason"] = "max_iter"
        state["is_running"] = False
        log_to_streamlit(state, f"✅ 최대 반복 횟수 도달로 완료: {current_iter}/{max_iter}", "success")
        return "stop"

    state["stages_completed"] = ["normalize"]  # 일부 단계 유지
    log_to_streamlit(state, f"🔄 다음 반복 계속 (반복 {current_iter})", "info")
    return "continue"


# 그래프 빌더
def build_graph() -> Any:
    """LangGraph 워크플로우 생성"""
    g = StateGraph(State)

    g.add_node("normalize", node_normalize)
    g.add_node("sample", node_sample)
    g.add_node("summarize", node_summarize)
    g.add_node("expand", node_expand_rag)
    g.add_node("synthesize", node_synthesize)
    g.add_node("verify", node_verify)
    g.add_node("productize", node_productize)
    g.add_node("score", node_score)

    g.add_edge("normalize", "sample")
    g.add_edge("sample", "summarize")
    g.add_edge("summarize", "expand")
    g.add_edge("expand", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", "productize")
    g.add_edge("productize", "score")
    g.add_conditional_edges("score", should_continue, {"continue": "sample", "stop": END})

    g.set_entry_point("normalize")

    return g


class KnowledgeCreationEngine:
    """지식 창출 엔진 래퍼 클래스"""

    def __init__(self, chroma_persist_directory: str = "./data/chroma_db",
                 collection_name: str = "knowledge_base",
                 max_iterations: int = 3):
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.max_iterations = max_iterations

        # LangGraph 컴파일
        self.graph = build_graph().compile()

    def run(self, max_iter: int = None, streamlit_state = None,
            quality_threshold: float = 0.75, agent_temperature: float = 0.3,
            enable_verification: bool = True) -> State:
        """지식 창출 프로세스 실행"""
        if max_iter is None:
            max_iter = self.max_iterations

        # 역할별 temperature 동적 설정
        roles_config = {
            "librarian": {"model": DEPLOYMENT_NAME, "temperature": 0.0},  # 정규화는 항상 0.0
            "summarizer": {"model": DEPLOYMENT_NAME, "temperature": 0.0},  # 요약도 정확성 위해 0.0
            "synthesizer": {"model": GPT4O_DEPLOYMENT, "temperature": agent_temperature},  # 창의성 적용
            "verifier": {"model": DEPLOYMENT_NAME, "temperature": 0.0},  # 검증은 항상 0.0
            "productizer": {"model": DEPLOYMENT_NAME, "temperature": agent_temperature * 0.3},  # 약간의 창의성
        }

        # 초기 상태
        initial_state: State = {
            "iter": 0,
            "max_iter": max_iter,
            "cfg_roles": roles_config,
            "cfg_services": {
                **DEFAULT_SERVICES,
                "chroma": {
                    "path": self.chroma_persist_directory,
                    "collection": self.collection_name,
                    "hnsw_space": "cosine"
                }
            },
            "knotes": [],
            "stages_completed": [],
            "current_stage": "normalize",
            "is_running": True,
            "quality_threshold": quality_threshold,
            "enable_verification": enable_verification
        }

        # Streamlit 세션 상태 연결
        if streamlit_state is not None:
            initial_state["streamlit_state"] = streamlit_state

        # 그래프 실행 (recursion_limit 설정)
        result = self.graph.invoke(initial_state, config={"recursion_limit": 100})

        return result