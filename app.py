import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import tempfile
import json
from typing import List, Dict
from knowledge.service import KnowledgeService, RAGService
from knowledge_creation.creation_engine import KnowledgeCreationEngine
from board.service import BoardService

# JSON 직렬화를 위한 커스텀 serializer
def json_serializer(obj):
    """JSON 직렬화를 위한 커스텀 serializer"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="AI Knowledge Management System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 네비게이션
def sidebar_navigation():
    st.sidebar.title("🧠 AI Knowledge System")
    st.sidebar.markdown("---")

    pages = {
        "지식등록": "📚",
        "게시판": "📋",
        "지식창출": "🔬"
    }

    selected_page = st.sidebar.radio(
        "메뉴 선택",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("AI 기반 지식관리 시스템")

    return selected_page

# 세션 상태 초기화
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'board_posts' not in st.session_state:
        st.session_state.board_posts = []
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = []
    if 'knowledge_service' not in st.session_state:
        st.session_state.knowledge_service = KnowledgeService()
    if 'board_service' not in st.session_state:
        st.session_state.board_service = BoardService()
    if 'rag_service' not in st.session_state:
        st.session_state.rag_service = RAGService()
    if 'creation_engine' not in st.session_state:
        st.session_state.creation_engine = KnowledgeCreationEngine()


# 지식등록 화면
def knowledge_registration_page():
    st.title("📚 지식 등록")
    st.markdown("문서를 업로드하여 AI가 분석하고 보완합니다.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 파일 업로드")
        uploaded_file = st.file_uploader(
            "문서를 선택하세요",
            type=['txt', 'pdf', 'docx', 'pptx'],
            help="PDF, Word, PowerPoint, 텍스트 파일을 지원합니다."
        )

        if uploaded_file is not None:
            st.success(f"파일 업로드 완료: {uploaded_file.name}")

            # 파일이 바뀌면 기존 분석 결과 클리어
            if 'current_filename' not in st.session_state or st.session_state.current_filename != uploaded_file.name:
                st.session_state.current_filename = uploaded_file.name
                if 'current_analysis' in st.session_state:
                    del st.session_state.current_analysis
                if 'current_file_content' in st.session_state:
                    del st.session_state.current_file_content

            # MarkItDown을 사용한 파일 전체 내용 표시
            knowledge_service = st.session_state.knowledge_service

            with st.spinner("파일 내용을 변환하는 중..."):
                success, full_content = knowledge_service.convert_file_to_text(uploaded_file)
                display_content = full_content if success else "파일 변환 실패"

            st.text_area("📄 파일 내용", display_content, height=400)

            # 전체 변환된 내용을 세션에 저장 (분석용)
            if 'current_file_content' not in st.session_state:
                success, full_content = knowledge_service.convert_file_to_text(uploaded_file)
                if success:
                    st.session_state.current_file_content = full_content
                else:
                    st.session_state.current_file_content = preview_content

            if st.button("🔍 분석 및 보완", type="primary"):
                if 'current_file_content' in st.session_state:
                    with st.spinner("AI가 문서를 분석하고 있습니다..."):
                        time.sleep(2)  # 분석 시뮬레이션

                        analysis_result = knowledge_service.analyze_document(
                            st.session_state.current_file_content,
                            uploaded_file.name
                        )
                        st.session_state.current_analysis = analysis_result
                        st.success("분석 완료!")
                else:
                    st.error("파일 내용을 먼저 변환해주세요.")

    with col2:
        st.subheader("📊 분석 결과")

        if 'current_analysis' in st.session_state:
            result = st.session_state.current_analysis

            # 품질 점수
            original_length = result.get('original_length', 0)
            enhanced_length = result.get('enhanced_length', 0)
            improvement_delta = enhanced_length - original_length

            col_score1, col_score2 = st.columns(2)
            with col_score1:
                st.metric("품질 점수", f"{result['quality_score']}점")
            with col_score2:
                st.metric("문서 길이", f"{enhanced_length}자", delta=f"+{improvement_delta}자")

            # 메타데이터 정보
            if 'metadata' in result:
                metadata = result['metadata']

                # 에러 정보가 있으면 경고 표시
                if 'error_info' in metadata:
                    error_info = metadata['error_info']
                    st.warning(f"⚠️ AI 분석 실패: {error_info['llm_error'][:100]}... (기본 분석기 사용됨)")

                with st.expander("📊 분석 메타데이터"):
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        st.write(f"**분석 시간:** {metadata['analyzed_at'].strftime('%Y-%m-%d %H:%M')}")
                    with col_meta2:
                        st.write(f"**파일명:** {metadata['filename']}")
                    with col_meta3:
                        analyzer_version = metadata['analyzer_version']
                        if 'Fallback' in analyzer_version:
                            st.write(f"**분석기 버전:** {analyzer_version} 🔄")
                        else:
                            st.write(f"**분석기 버전:** {analyzer_version} ✨")

                    # LLM 메타데이터가 있으면 추가 표시
                    if 'llm_metadata' in metadata and metadata['llm_metadata']:
                        llm_meta = metadata['llm_metadata']
                        st.write("**AI 분석 정보:**")
                        if 'type' in llm_meta:
                            st.write(f"- 문서 종류: {llm_meta['type']}")
                        if 'project_area' in llm_meta:
                            st.write(f"- 프로젝트 분야: {llm_meta['project_area']}")
                        if 'keywords' in llm_meta and llm_meta['keywords']:
                            st.write(f"- 키워드: {', '.join(llm_meta['keywords'])}")

            # 발견된 문제점
            st.subheader("🔍 발견된 문제점")
            for issue in result.get('issues_found', []):
                st.warning(f"• {issue}")

            # 개선 사항
            st.subheader("✨ 제안 개선사항")
            for improvement in result.get('improvements', []):
                st.info(f"• {improvement}")

            # 보완된 지식 문서 생성 버튼
            if st.button("📄 보완된 지식 문서 생성", type="primary", key="generate_enhanced_doc"):
                with st.spinner("AI가 보완된 지식 문서를 생성하고 있습니다..."):
                    enhanced_result = knowledge_service.generate_enhanced_knowledge_document(
                        result, uploaded_file.name
                    )
                    st.session_state.enhanced_document = enhanced_result
                    st.success("보완된 지식 문서 생성 완료!")

            # 보완된 문서가 생성되었을 때만 저장 버튼 표시
            if 'enhanced_document' in st.session_state:
                st.markdown("---")
                st.subheader("📤 문서 저장")

                # 통합 저장 버튼
                if st.button("💾 VectorDB & 게시판에 저장", type="primary", key="save_all"):
                    with st.spinner("VectorDB와 게시판에 저장하고 있습니다..."):
                        # 서비스 클래스 인스턴스 가져오기
                        rag_service = st.session_state.rag_service
                        knowledge_service = st.session_state.knowledge_service

                        # Azure Blob Storage에 파일 업로드
                        # 1. 원본 파일 업로드
                        uploaded_file.seek(0)  # 파일 포인터 리셋
                        original_url = knowledge_service.file_processor.upload_file(uploaded_file, "original")

                        # 2. 보완된 문서 업로드 ({원본파일명}_enhanced.md)
                        import io
                        original_filename = uploaded_file.name.rsplit('.', 1)[0]  # 확장자 제거
                        enhanced_filename = f"{original_filename}_enhanced.md"
                        enhanced_content = st.session_state.enhanced_document['enhanced_content']

                        # 보완 문서를 파일 형태로 변환
                        enhanced_file = io.BytesIO(enhanced_content.encode('utf-8'))
                        enhanced_file.name = enhanced_filename
                        enhanced_url = knowledge_service.file_processor.upload_file(enhanced_file, "enhanced")

                        # RAGService로 VectorDB에 임베딩 및 저장
                        vector_result = rag_service.embed_and_store(
                            text=enhanced_content,
                            metadata={
                                "filename": uploaded_file.name,
                                "enhanced_filename": enhanced_filename,
                                "original_url": original_url,
                                "enhanced_url": enhanced_url,
                                "quality_score": st.session_state.enhanced_document.get('quality_score', 0)
                            },
                            split_type="semantic"
                        )

                        # RAGService로 게시판 DB에 저장
                        board_result = rag_service.save_to_board_db(
                            title=f"[AI 보완] {uploaded_file.name}",
                            content=enhanced_content,
                            enhanced_doc_url=enhanced_url,
                            original_doc_url=original_url,
                            quality_score=st.session_state.enhanced_document.get('quality_score', 0),
                            metadata={
                                "filename": uploaded_file.name,
                                "generation_metadata": st.session_state.enhanced_document.get('generation_metadata', {})
                            }
                        )

                        # 결과 표시
                        col_result1, col_result2 = st.columns(2)

                        with col_result1:
                            if vector_result['success']:
                                st.success(f"✅ {vector_result['message']} ({vector_result['chunk_count']}개 청크)")
                            else:
                                st.warning(f"⚠️ VectorDB: {vector_result['message']}")

                        with col_result2:
                            if board_result['success']:
                                st.success(f"✅ {board_result['message']}")
                            else:
                                st.warning(f"⚠️ 게시판: {board_result['message']}")

                # 생성된 보완 문서 보기
                with st.expander("📝 생성된 보완 문서 보기"):
                    st.markdown(st.session_state.enhanced_document['enhanced_content'])

# 게시판 화면
def board_page():
    st.title("📋 지식 게시판")
    st.markdown("AI가 분석하고 보완한 지식들을 확인할 수 있습니다.")

    # RAGService에서 게시글 가져오기
    rag_service = st.session_state.rag_service
    board_posts = rag_service.get_board_posts(limit=100)

    if not board_posts:
        st.info("아직 등록된 게시글이 없습니다. 지식등록 메뉴에서 문서를 업로드해주세요.")
        return

    # 게시글 목록
    st.subheader(f"📚 총 {len(board_posts)}개의 지식이 등록되었습니다")

    for i, post in enumerate(board_posts):
        with st.expander(f"📄 {post['title']} (품질점수: {post['quality_score']}점)"):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

            with col1:
                st.write(f"**작성자:** {post['author']}")
            with col2:
                st.write(f"**등록일:** {post['created_at'][:10]}")
            with col3:
                st.write(f"**조회수:** {post['views']}")
            with col4:
                if st.button("조회", key=f"view_{i}"):
                    # 조회수 증가 (get_board_post_by_id가 자동으로 처리)
                    rag_service.get_board_post_by_id(post['id'])
                    st.rerun()

            st.markdown("---")

            # 다운로드 링크가 있으면 표시
            if post.get('enhanced_doc_url') or post.get('original_doc_url'):
                col_link1, col_link2 = st.columns(2)
                with col_link1:
                    if post.get('original_doc_url'):
                        st.markdown(f"[📥 원본 문서 다운로드]({post['original_doc_url']})")
                with col_link2:
                    if post.get('enhanced_doc_url'):
                        st.markdown(f"[📥 보완 문서 다운로드]({post['enhanced_doc_url']})")
                st.markdown("---")

            st.text_area("내용", post['content'], height=200, key=f"content_{i}")

# 지식창출 화면
def knowledge_creation_page():
    st.title("🔬 지식 창출")
    st.markdown("Multi-Agent 시스템을 통해 새로운 지식을 창출합니다.")

    # 상단 설정 패널
    with st.expander("⚙️ 창출 설정", expanded=False):
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

        with col_cfg1:
            st.subheader("🎯 창출 목표")
            creation_goal = st.selectbox(
                "목표 선택",
                ["패턴 발견", "아날로지 생성", "지식 융합", "혁신 아이디어"]
            )
            quality_threshold = st.slider("품질 임계값", 0.5, 1.0, 0.75, 0.05)

        with col_cfg2:
            st.subheader("🤖 Agent 설정")
            max_iterations = st.slider("최대 반복 횟수", 1, 10, 3)
            agent_temperature = st.slider("창의성 수준", 0.0, 1.0, 0.3, 0.1)
            enable_verification = st.checkbox("검증 단계 활성화", True)

        with col_cfg3:
            st.subheader("📊 데이터 소스")
            vector_db_size = len(st.session_state.get('vector_db', []))
            board_size = len(st.session_state.get('board_posts', []))
            st.metric("VectorDB 문서", f"{vector_db_size}개")
            st.metric("게시판 문서", f"{board_size}개")

            include_vectordb = st.checkbox("VectorDB 포함", True)
            include_board = st.checkbox("게시판 포함", True)

    # 메인 컨테이너
    col_main1, col_main2 = st.columns([2, 1])

    with col_main1:
        # 창출 프로세스 시작 버튼
        st.subheader("🚀 지식 창출 실행")

        if st.button("🔬 Multi-Agent 지식 창출 시작", type="primary", key="start_creation"):
            try:
                # 프로그레스 바와 상태 표시 컨테이너
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                stage_detail_placeholder = st.empty()

                # 진행 상황 컨테이너
                with st.status("🚀 지식 창출 진행 중...", expanded=True) as status:
                    stages = ["normalize", "sample", "summarize", "expand", "synthesize", "verify", "productize", "score"]
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

                    # 진행 상황을 표시할 placeholder
                    log_container = st.container()

                    # 세션 상태 초기화
                    if 'creation_state' not in st.session_state:
                        st.session_state.creation_state = {}

                    if 'creation_logs' not in st.session_state:
                        st.session_state.creation_logs = []

                    st.session_state.creation_state.update({
                        "is_running": True,
                        "current_stage": "normalize",
                        "stages_completed": [],
                        "iteration": 0,
                        "max_iterations": max_iterations
                    })

                    # 실시간 진행 상황 표시 영역
                    st.write("---")
                    st.write("**📝 실행 과정:**")

                    # 각 단계별 placeholder 생성
                    stage_placeholders = {}
                    for stage in stages:
                        stage_placeholders[stage] = st.empty()
                        stage_placeholders[stage].info(f"⏳ {stage_names[stage]} - 대기 중...")

                    # 상세 로그 영역
                    st.write("---")
                    st.write("**🔍 상세 로그:**")
                    log_placeholder = st.empty()

                    # LangGraph 엔진 실행 (Streamlit 세션 상태 전달)
                    creation_engine = st.session_state.creation_engine

                    # 세션 상태에 placeholder 저장
                    st.session_state.stage_placeholders = stage_placeholders
                    st.session_state.log_placeholder = log_placeholder

                    result = creation_engine.run(
                        max_iter=max_iterations,
                        streamlit_state=st.session_state,
                        quality_threshold=quality_threshold,
                        agent_temperature=agent_temperature,
                        enable_verification=enable_verification
                    )

                    # 실행 후 로그 표시
                    with log_placeholder.container():
                        if hasattr(st.session_state, 'creation_logs') and st.session_state.creation_logs:
                            for log in st.session_state.creation_logs[-50:]:  # 최근 50개만 표시
                                if log['level'] == 'success':
                                    st.success(log['message'])
                                elif log['level'] == 'warning':
                                    st.warning(log['message'])
                                elif log['level'] == 'error':
                                    st.error(log['message'])
                                else:
                                    st.info(log['message'])
                        else:
                            st.write("로그가 없습니다.")

                    status.update(label="✅ 지식 창출 완료!", state="complete", expanded=False)

                # 결과 요약
                st.markdown("---")
                st.markdown("### 📊 실행 결과 요약")

                # 단계별 완료 상태 표시
                st.markdown("#### 🔄 완료된 단계")
                completed_stages = result.get('stages_completed', [])

                cols = st.columns(len(stages))
                for i, (stage, name) in enumerate(stage_names.items()):
                    with cols[i]:
                        if stage in completed_stages:
                            st.success(f"✅")
                            st.caption(name.split()[1])  # 이모지 제외한 이름
                        else:
                            st.warning(f"⏸️")
                            st.caption(name.split()[1])

                st.markdown("---")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("샘플 수", len(result.get('samples', [])))
                with col2:
                    st.metric("제안 수", len(result.get('proposals', [])))
                with col3:
                    accepted = sum(1 for v in result.get('verdicts', []) if v.get('verdict') == 'accept')
                    st.metric("승인된 제안", f"{accepted}/{len(result.get('verdicts', []))}")
                with col4:
                    st.metric("K-Note 생성", len(result.get('knotes', [])))

                # 품질 점수
                if result.get('scores'):
                    st.markdown("### 📈 품질 평가 점수")
                    scores = result['scores']
                    score_col1, score_col2, score_col3, score_col4 = st.columns(4)

                    with score_col1:
                        st.metric("신규성", f"{scores.get('novelty', 0):.2f}")
                    with score_col2:
                        st.metric("커버리지", f"{scores.get('coverage', 0):.2f}")
                    with score_col3:
                        st.metric("유용성", f"{scores.get('utility', 0):.2f}")
                    with score_col4:
                        import numpy as np
                        avg = np.mean(list(scores.values()))
                        st.metric("평균", f"{avg:.2f}")

                # 세션 상태 업데이트
                st.session_state.creation_state = {
                    "current_stage": result.get("current_stage", "score"),
                    "iteration": result.get("iter", max_iterations),
                    "max_iterations": max_iterations,
                    "stages_completed": result.get("stages_completed", []),
                    "current_samples": [f"Sample_{i+1}" for i, s in enumerate(result.get("samples", []))],
                    "summaries": [f"Summary_{i+1}" for i, s in enumerate(result.get("summaries", []))],
                    "proposals": result.get("proposals", []),
                    "verdicts": result.get("verdicts", []),
                    "knotes": result.get("knotes", []),
                    "scores": result.get("scores", {}),
                    "stop_reason": result.get("stop_reason"),
                    "is_running": False
                }

                if result.get("knotes"):
                    st.success(f"✅ {len(result['knotes'])}개의 K-Note가 성공적으로 생성되었습니다!")
                else:
                    st.warning(f"⚠️ K-Note가 생성되지 않았습니다. 종료 이유: {result.get('stop_reason', 'unknown')}")

                st.rerun()
            except Exception as e:
                st.error(f"지식 창출 중 오류 발생: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        # 프로세스 결과 요약 표시
        if 'creation_state' in st.session_state:
            st.markdown("---")
            show_process_summary()

    # 결과 표시 영역
    if 'creation_state' in st.session_state:
        show_creation_results()

def show_process_summary():
    """완료된 프로세스 결과 요약"""
    st.subheader("🔄 Multi-Agent 워크플로우 결과")

    creation_state = st.session_state.creation_state

    # 완료된 단계들 표시
    with st.container():
        stages_info = {
            "normalize": ("📚 Librarian", "데이터 정규화"),
            "sample": ("🎲 Sampler", "다양성 샘플링"),
            "summarize": ("📝 Summarizer", "구조화 요약"),
            "expand": ("🔍 Expander", "RAG 컨텍스트 확장"),
            "synthesize": ("🧬 Synthesizer", "지식 융합"),
            "verify": ("✅ Verifier", "검증"),
            "productize": ("📋 Productizer", "K-Note 생성"),
            "score": ("📊 Evaluator", "평가")
        }

        for stage, (icon_name, desc) in stages_info.items():
            with st.expander(f"{icon_name} - {desc}", expanded=False):
                if stage in creation_state.get("stages_completed", []):
                    st.success(f"✅ {desc} 완료")

                    # 단계별 세부 정보 표시
                    if stage == "sample" and creation_state.get("current_samples"):
                        st.write(f"- 선택된 샘플: {len(creation_state['current_samples'])}개")
                    elif stage == "summarize" and creation_state.get("summaries"):
                        st.write(f"- 생성된 요약: {len(creation_state['summaries'])}개")
                    elif stage == "synthesize" and creation_state.get("proposals"):
                        st.write(f"- 생성된 제안: {len(creation_state['proposals'])}개")
                    elif stage == "verify" and creation_state.get("verdicts"):
                        accepted = sum(1 for v in creation_state["verdicts"] if v.get("verdict") == "accept")
                        st.write(f"- 검증된 제안: {len(creation_state['verdicts'])}개 (승인: {accepted}개)")
                    elif stage == "productize" and creation_state.get("knotes"):
                        st.write(f"- 생성된 K-Note: {len(creation_state['knotes'])}개")
                    elif stage == "score" and creation_state.get("scores"):
                        scores = creation_state["scores"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("신규성", f"{scores.get('novelty', 0):.2f}")
                        with col2:
                            st.metric("커버리지", f"{scores.get('coverage', 0):.2f}")
                        with col3:
                            st.metric("유용성", f"{scores.get('utility', 0):.2f}")
                else:
                    st.info("⏳ 미완료")

def show_creation_results():
    """창출 결과 표시"""
    if 'creation_state' not in st.session_state:
        return
        
    creation_state = st.session_state.creation_state

    # 디버깅 정보 표시 (개발용)
    if st.checkbox("🔍 디버깅 정보 표시", False):
        st.write("**Creation State Debug:**")
        st.write(f"- is_running: {creation_state.get('is_running')}")
        st.write(f"- knotes count: {len(creation_state.get('knotes', []))}")
        st.write(f"- knotes type: {type(creation_state.get('knotes', []))}")
        if creation_state.get('knotes'):
            st.write(f"- first knote type: {type(creation_state['knotes'][0])}")
        st.write("**Full creation_state keys:**")
        st.write(list(creation_state.keys()))

    # K-Note가 있으면 표시 (is_running 조건 완화)
    knotes = creation_state.get("knotes", [])
    if knotes and len(knotes) > 0:
        st.markdown("---")
        st.subheader("🎉 지식 창출 결과")

        # 최종 성과 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("생성된 K-Note", f"{len(creation_state.get('knotes', []))}개")
        with col2:
            st.metric("처리된 샘플", f"{len(creation_state.get('current_samples', []))}개")
        with col3:
            st.metric("반복 횟수", f"{creation_state.get('iteration', 0)}회")
        with col4:
            scores = creation_state.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            st.metric("평균 품질", f"{avg_score:.2f}")

        # 생성된 K-Note 목록
        st.subheader("📋 생성된 K-Note")
        knotes = creation_state.get("knotes", [])

        if isinstance(knotes, list) and len(knotes) > 0:
            for i, knote in enumerate(knotes, 1):
                # K-Note가 딕셔너리인지 확인
                if isinstance(knote, dict):
                    k_note_id = knote.get("k_note_id", f"K-Note {i}")
                    title = knote.get("title", "제목 없음")
                    proposal = knote.get("proposal", "")
                    evidence = knote.get("evidence", [])
                    status = knote.get('status', 'draft')
                    version = knote.get('version', '1.0')

                    # 상태에 따른 아이콘
                    status_icon = "✅" if status == "validated" else "📝"

                    with st.expander(f"{status_icon} **{k_note_id}**: {title}", expanded=True):
                        # 헤더 정보
                        header_col1, header_col2, header_col3 = st.columns(3)
                        with header_col1:
                            st.markdown(f"**📌 상태**: `{status}`")
                        with header_col2:
                            st.markdown(f"**🔢 버전**: `{version}`")
                        with header_col3:
                            owners = knote.get('owners', [])
                            if owners:
                                st.markdown(f"**👥 담당자**: {', '.join(owners)}")

                        st.markdown("---")

                        # 제안 내용
                        st.markdown("### 💡 핵심 제안")
                        st.info(proposal)

                        # 적용 가능성
                        if "applicability" in knote and isinstance(knote["applicability"], dict):
                            st.markdown("### 🎯 적용 가능성")
                            applicability = knote["applicability"]

                            col_app1, col_app2 = st.columns(2)

                            with col_app1:
                                if "when" in applicability and isinstance(applicability['when'], list) and applicability['when']:
                                    st.markdown("**✅ 적용 권장 상황**")
                                    for item in applicability['when']:
                                        st.markdown(f"- {item}")

                                if "assumptions" in applicability and isinstance(applicability.get('assumptions'), list) and applicability['assumptions']:
                                    st.markdown("**📋 전제 조건**")
                                    for assumption in applicability['assumptions']:
                                        st.markdown(f"- {assumption}")

                            with col_app2:
                                if "when_not" in applicability and isinstance(applicability['when_not'], list) and applicability['when_not']:
                                    st.markdown("**❌ 적용 제외 상황**")
                                    for item in applicability['when_not']:
                                        st.markdown(f"- {item}")

                        # 예상 효과
                        if "metrics_effect" in knote and knote["metrics_effect"]:
                            st.markdown("### 📈 예상 효과")
                            metrics = knote["metrics_effect"]
                            if isinstance(metrics, dict):
                                # 유효한 메트릭만 필터링 (숫자 또는 문자열만)
                                valid_metrics = {}
                                list_metrics = {}

                                for key, value in metrics.items():
                                    if isinstance(value, (int, float, str)):
                                        # 리스트가 아닌 값만 메트릭으로 표시
                                        valid_metrics[key] = value
                                    elif isinstance(value, list):
                                        # 리스트는 별도로 처리
                                        list_metrics[key] = value

                                # 메트릭 카드로 표시 (숫자/문자열)
                                if valid_metrics:
                                    metric_cols = st.columns(len(valid_metrics))
                                    for idx, (key, value) in enumerate(valid_metrics.items()):
                                        with metric_cols[idx]:
                                            st.metric(key, value)

                                # 리스트 형태는 별도 표시
                                if list_metrics:
                                    for key, items in list_metrics.items():
                                        st.markdown(f"**{key}**")
                                        for item in items:
                                            st.markdown(f"- {item}")
                            elif isinstance(metrics, list):
                                # 리스트인 경우
                                for item in metrics:
                                    st.markdown(f"- {item}")
                            else:
                                st.write(str(metrics))

                        # 근거 문서
                        if evidence and isinstance(evidence, list) and len(evidence) > 0:
                            st.markdown("### 📚 근거 문서")
                            for idx, ev in enumerate(evidence[:5], 1):
                                if isinstance(ev, dict):
                                    doc_id = ev.get("doc_id", "unknown")
                                    chunk_id = ev.get("chunk_id", "")
                                    quote = ev.get("quote", "")
                                    confidence = ev.get("confidence", 0)

                                    with st.container():
                                        confidence_pct = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else str(confidence)
                                        st.markdown(f"**{idx}. 문서 ID**: `{doc_id}` | **신뢰도**: {confidence_pct}")
                                        if quote:
                                            st.caption(f"💬 \"{quote[:200]}...\"" if len(quote) > 200 else f"💬 \"{quote}\"")
                                else:
                                    st.markdown(f"- {str(ev)}")

                        # 위험 및 제한사항
                        if "risks_limits" in knote and isinstance(knote["risks_limits"], list) and knote["risks_limits"]:
                            st.markdown("### ⚠️ 위험 및 제한사항")
                            for risk in knote["risks_limits"]:
                                st.warning(f"⚠️ {str(risk)}")

                        # 권장 실험
                        if "recommended_experiments" in knote and isinstance(knote["recommended_experiments"], list) and knote["recommended_experiments"]:
                            st.markdown("### 🧪 권장 실험/검증")
                            for exp_idx, exp in enumerate(knote["recommended_experiments"], 1):
                                if isinstance(exp, dict):
                                    exp_name = exp.get("name", f"실험 {exp_idx}")
                                    exp_duration = exp.get("duration", "미정")
                                    exp_criteria = exp.get("success_criteria", "미정")

                                    st.markdown(f"**{exp_idx}. {exp_name}**")
                                    st.markdown(f"- **기간**: {exp_duration}")
                                    st.markdown(f"- **성공 기준**: {exp_criteria}")
                                else:
                                    st.markdown(f"{exp_idx}. {str(exp)}")

                        # 관련 K-Note
                        if "related" in knote and isinstance(knote["related"], list) and knote["related"]:
                            st.markdown("### 🔗 관련 K-Note")
                            st.write(", ".join([f"`{r}`" for r in knote["related"]]))

                        st.markdown("---")

                        # 구체화 단계
                        st.markdown("### 📝 문서 구체화")
                        
                        # 구체화 상태 확인
                        enhanced_key = f"enhanced_doc_{k_note_id}"
                        
                        if enhanced_key not in st.session_state:
                            # 구체화 버튼
                            col_enhance1, col_enhance2 = st.columns([2, 1])
                            with col_enhance1:
                                additional_points = st.text_area(
                                    "추가 보완사항 (선택사항)", 
                                    placeholder="구체화 과정에서 추가로 보완하고 싶은 내용을 입력하세요...",
                                    key=f"additional_points_{i}",
                                    height=100
                                )
                            with col_enhance2:
                                st.markdown("<br>", unsafe_allow_html=True)
                                if st.button(f"🔄 문서 구체화", key=f"enhance_knote_{i}", type="primary", use_container_width=True):
                                    with st.spinner("K-Note를 표준 문서로 구체화하는 중..."):
                                        try:
                                            # 디버깅: K-Note 타입 확인
                                            if not isinstance(knote, dict):
                                                st.error(f"K-Note 타입 오류: 딕셔너리가 아닌 {type(knote)} 타입입니다.")
                                                st.write("K-Note 내용:", knote)
                                                return
                                            
                                            knowledge_service = st.session_state.knowledge_service
                                            enhanced_result = knowledge_service.enhance_knote_to_standard_document(
                                                knote, additional_points
                                            )
                                            st.session_state[enhanced_key] = enhanced_result
                                            st.success("✅ 문서 구체화가 완료되었습니다!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"구체화 중 오류 발생: {str(e)}")
                                            # 추가 디버깅 정보
                                            st.write("K-Note 타입:", type(knote))
                                            st.write("K-Note 내용 (처음 500자):", str(knote)[:500])
                        else:
                            # 구체화된 문서 표시
                            enhanced_result = st.session_state[enhanced_key]
                            
                            st.success("✅ 구체화된 표준 문서가 생성되었습니다!")
                            
                            # 구체화된 문서 미리보기
                            with st.expander("📄 구체화된 문서 미리보기", expanded=False):
                                st.markdown(enhanced_result['enhanced_content'])
                            
                            # 품질 점수 표시
                            quality_score = enhanced_result.get('quality_score', 0)
                            st.metric("문서 품질 점수", f"{quality_score}/100")
                            
                            st.markdown("---")
                            
                            # VectorDB & 게시판 저장 버튼
                            st.markdown("### 💾 지식 저장")
                            
                            col_save1, col_save2, col_save3 = st.columns(3)
                            
                            with col_save1:
                                if st.button(f"📚 VectorDB & 게시판에 저장", key=f"save_enhanced_{i}", type="primary", use_container_width=True):
                                    with st.spinner("Azure Blob Storage, VectorDB, 게시판에 저장하는 중..."):
                                        try:
                                            # 서비스 초기화
                                            rag_service = RAGService()
                                            knowledge_service = st.session_state.knowledge_service
                                            
                                            # 1. Azure Blob Storage에 구체화된 문서 업로드
                                            enhanced_url = knowledge_service.upload_enhanced_document_to_blob(
                                                enhanced_result['enhanced_content'],
                                                k_note_id,
                                                title
                                            )
                                            
                                            # 메타데이터 준비 (Azure URL 포함)
                                            metadata = {
                                                "doc_id": k_note_id,
                                                "title": title,
                                                "author": ', '.join(knote.get('owners', ['AI Knowledge System'])),
                                                "quality_score": quality_score,
                                                "source_type": "enhanced_knote",
                                                "k_note_id": k_note_id,
                                                "version": version,
                                                "status": status,
                                                "enhanced_url": enhanced_url
                                            }
                                            
                                            # 2. VectorDB에 저장
                                            vector_result = rag_service.embed_and_store(
                                                text=enhanced_result['enhanced_content'],
                                                metadata=metadata,
                                                split_type="semantic"
                                            )
                                            
                                            # 3. 게시판에 저장
                                            board_result = rag_service.save_to_board_db(
                                                title=title,
                                                content=enhanced_result['enhanced_content'],
                                                enhanced_doc_url=enhanced_url,
                                                author=', '.join(knote.get('owners', ['AI Knowledge System'])),
                                                quality_score=quality_score,
                                                metadata={
                                                    **enhanced_result.get('generation_metadata', {}),
                                                    "original_knote": knote,
                                                    "enhanced_url": enhanced_url
                                                }
                                            )
                                            
                                            # 결과 확인 및 표시
                                            success_messages = []
                                            error_messages = []
                                            
                                            if enhanced_url:
                                                success_messages.append("Azure Blob Storage 업로드")
                                            else:
                                                error_messages.append("Azure Blob Storage 업로드 실패")
                                            
                                            if vector_result['success']:
                                                success_messages.append(f"VectorDB: {vector_result['chunk_count']}개 청크")
                                            else:
                                                error_messages.append(f"VectorDB: {vector_result['message']}")
                                            
                                            if board_result['success']:
                                                success_messages.append("게시판: 1개 게시글")
                                            else:
                                                error_messages.append(f"게시판: {board_result['message']}")
                                            
                                            if not error_messages:
                                                st.success(f"✅ 저장 완료! {', '.join(success_messages)}")
                                                # 저장 완료 표시
                                                st.session_state[f"saved_{k_note_id}"] = True
                                                # 업로드된 URL 정보 표시
                                                if enhanced_url:
                                                    st.info(f"📁 Azure Blob Storage URL: {enhanced_url}")
                                            else:
                                                st.error(f"저장 실패: {', '.join(error_messages)}")
                                                if success_messages:
                                                    st.warning(f"부분 성공: {', '.join(success_messages)}")
                                                
                                        except Exception as e:
                                            st.error(f"저장 중 오류 발생: {str(e)}")
                                            import traceback
                                            st.code(traceback.format_exc())
                            
                            with col_save2:
                                if st.button(f"🔄 재구체화", key=f"re_enhance_{i}", use_container_width=True):
                                    # 구체화 상태 초기화
                                    del st.session_state[enhanced_key]
                                    st.rerun()
                            
                            with col_save3:
                                # JSON 다운로드
                                enhanced_json = {
                                    "original_knote": knote,
                                    "enhanced_document": enhanced_result
                                }
                                json_str = json.dumps(enhanced_json, ensure_ascii=False, indent=2, default=json_serializer)
                                st.download_button(
                                    label="📥 전체 다운로드",
                                    data=json_str,
                                    file_name=f"{k_note_id}_enhanced.json",
                                    mime="application/json",
                                    key=f"download_enhanced_{i}",
                                    use_container_width=True
                                )
                            
                            # 저장 완료 표시
                            if st.session_state.get(f"saved_{k_note_id}"):
                                st.success("✅ 이 문서는 이미 VectorDB와 게시판에 저장되었습니다.")
                        
                        # 기본 액션 버튼 (구체화 전에도 사용 가능)
                        st.markdown("---")
                        st.markdown("### 🔧 기본 액션")
                        col_basic1, col_basic2 = st.columns(2)
                        
                        with col_basic1:
                            # 원본 K-Note JSON 다운로드
                            json_str = json.dumps(knote, ensure_ascii=False, indent=2, default=json_serializer)
                            st.download_button(
                                label="📥 원본 K-Note 다운로드",
                                data=json_str,
                                file_name=f"{k_note_id}_original.json",
                                mime="application/json",
                                key=f"download_original_{i}",
                                use_container_width=True
                            )
                        
                        with col_basic2:
                            if st.button(f"🗑️ 이 K-Note 삭제", key=f"delete_knote_{i}", use_container_width=True):
                                # K-Note 삭제 (세션에서 제거)
                                if 'creation_state' in st.session_state and 'knotes' in st.session_state.creation_state:
                                    knotes_list = st.session_state.creation_state['knotes']
                                    if i-1 < len(knotes_list):
                                        del knotes_list[i-1]
                                        st.success("K-Note가 삭제되었습니다.")
                                        st.rerun()
                else:
                    # 문자열인 경우 (기존 시뮬레이션 형식)
                    with st.expander(f"K-Note {i}: {knote}"):
                        st.write("**제목**: 새로운 지식 패턴 발견")
                        st.text_area("내용", str(knote), height=100, key=f"knote_{i}")
    else:
        # K-Note가 없는 경우
        if creation_state.get("is_running"):
            st.info("🔄 지식 창출이 진행 중입니다...")
        elif creation_state.get("stages_completed"):
            st.warning(f"⚠️ 지식 창출이 완료되었지만 K-Note가 생성되지 않았습니다. 종료 이유: {creation_state.get('stop_reason', 'unknown')}")
            st.info("💡 팁: 품질 임계값을 낮추거나 최대 반복 횟수를 늘려보세요.")
        else:
            st.info("📝 지식 창출을 시작하면 결과가 여기에 표시됩니다.")

# 메인 앱
def main():
    initialize_session_state()

    # 사이드바 네비게이션
    selected_page = sidebar_navigation()

    # 페이지 라우팅
    if selected_page == "지식등록":
        knowledge_registration_page()
    elif selected_page == "게시판":
        board_page()
    elif selected_page == "지식창출":
        knowledge_creation_page()

if __name__ == "__main__":
    main()