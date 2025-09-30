import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import tempfile
from typing import List, Dict
from knowledge.service import KnowledgeService
from chatbot.service import ChatbotService
from board.service import BoardService

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
        "챗봇": "💬",
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
    if 'chatbot_service' not in st.session_state:
        st.session_state.chatbot_service = ChatbotService()
    if 'board_service' not in st.session_state:
        st.session_state.board_service = BoardService()


# 챗봇 화면
def chatbot_page():
    st.title("💬 AI 챗봇")
    st.markdown("저장된 지식을 바탕으로 질문에 답변드립니다.")

    # 채팅 히스토리 표시
    chat_container = st.container()

    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

    # 채팅 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # AI 응답 시뮬레이션
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            # 응답 생성 시뮬레이션
            if st.session_state.vector_db:
                response = f"저장된 지식을 바탕으로 답변드리겠습니다.\n\n'{prompt}'에 대한 답변:\n\n현재 {len(st.session_state.vector_db)}개의 문서가 데이터베이스에 저장되어 있으며, 관련 정보를 검색하여 답변을 생성했습니다."
            else:
                response = "아직 저장된 지식이 없습니다. 지식등록 메뉴에서 문서를 업로드해주세요."

            st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

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

            # MarkItDown을 사용한 파일 미리보기
            knowledge_service = st.session_state.knowledge_service

            with st.spinner("파일 내용을 변환하는 중..."):
                preview_content = knowledge_service.get_file_preview(uploaded_file)

            st.text_area("📄 파일 미리보기", preview_content, height=200)

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
                        chatbot_service = st.session_state.chatbot_service
                        board_service = st.session_state.board_service
                        knowledge_service = st.session_state.knowledge_service

                        # Azure Blob Storage에 파일 업로드
                        # 1. 원본 파일 업로드
                        uploaded_file.seek(0)  # 파일 포인터 리셋
                        knowledge_service.file_processor.upload_file(uploaded_file, "original")

                        # 2. 보완된 문서 업로드 ({원본파일명}_enhanced.md)
                        import io
                        original_filename = uploaded_file.name.rsplit('.', 1)[0]  # 확장자 제거
                        enhanced_filename = f"{original_filename}_enhanced.md"
                        enhanced_content = st.session_state.enhanced_document['enhanced_content']

                        # 보완 문서를 파일 형태로 변환
                        enhanced_file = io.BytesIO(enhanced_content.encode('utf-8'))
                        enhanced_file.name = enhanced_filename
                        knowledge_service.file_processor.upload_file(enhanced_file, "enhanced")

                        # VectorDB 저장
                        vector_result = chatbot_service.save_to_vector_db(
                            st.session_state.enhanced_document,
                            uploaded_file.name
                        )

                        # 게시판 저장
                        board_result = board_service.save_enhanced_document_to_board(
                            st.session_state.enhanced_document,
                            uploaded_file.name
                        )

                        # 결과 표시
                        col_result1, col_result2 = st.columns(2)

                        with col_result1:
                            if vector_result['success']:
                                st.success(f"✅ {vector_result['message']} (총 {vector_result['count']}개)")
                            else:
                                st.warning(f"⚠️ VectorDB: {vector_result['message']}")

                        with col_result2:
                            if board_result['success']:
                                st.success(f"✅ {board_result['message']} (총 {board_result['count']}개)")
                            else:
                                st.warning(f"⚠️ 게시판: {board_result['message']}")

                # 생성된 보완 문서 보기
                with st.expander("📝 생성된 보완 문서 보기"):
                    st.markdown(st.session_state.enhanced_document['enhanced_content'])

# 게시판 화면
def board_page():
    st.title("📋 지식 게시판")
    st.markdown("AI가 분석하고 보완한 지식들을 확인할 수 있습니다.")

    if not st.session_state.board_posts:
        st.info("아직 등록된 게시글이 없습니다. 지식등록 메뉴에서 문서를 업로드해주세요.")
        return

    # 게시글 목록
    st.subheader(f"📚 총 {len(st.session_state.board_posts)}개의 지식이 등록되었습니다")

    for i, post in enumerate(reversed(st.session_state.board_posts)):
        with st.expander(f"📄 {post['title']} (품질점수: {post['quality_score']}점)"):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

            with col1:
                st.write(f"**작성자:** {post['author']}")
            with col2:
                st.write(f"**등록일:** {post['timestamp'].strftime('%Y-%m-%d')}")
            with col3:
                st.write(f"**조회수:** {post['views']}")
            with col4:
                if st.button("조회", key=f"view_{i}"):
                    post['views'] += 1
                    st.rerun()

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
            # 세션 상태 초기화
            st.session_state.creation_state = {
                "current_stage": "normalize",
                "iteration": 1,
                "max_iterations": max_iterations,
                "stages_completed": [],
                "current_samples": [],
                "summaries": [],
                "proposals": [],
                "verdicts": [],
                "knotes": [],
                "scores": {},
                "is_running": True
            }
            st.rerun()

        # 프로세스 진행 상황 표시
        if 'creation_state' in st.session_state and st.session_state.creation_state.get('is_running'):
            st.markdown("---")
            show_creation_process()

    with col_main2:
        # 실시간 모니터링 패널
        st.subheader("📈 실시간 모니터링")

        if 'creation_state' in st.session_state:
            creation_state = st.session_state.creation_state

            # 진행률 표시
            stages = ["normalize", "sample", "summarize", "synthesize", "verify", "productize", "score"]
            current_stage_idx = stages.index(creation_state.get("current_stage", "normalize"))
            progress = (current_stage_idx + 1) / len(stages)

            st.progress(progress)
            st.write(f"**현재 단계:** {creation_state.get('current_stage', 'None')}")
            st.write(f"**반복:** {creation_state.get('iteration', 0)}/{creation_state.get('max_iterations', 0)}")

            # 단계별 상태
            with st.container():
                st.write("**단계별 진행 상황:**")
                for i, stage in enumerate(stages):
                    if stage in creation_state.get('stages_completed', []):
                        st.write(f"✅ {stage.title()}")
                    elif stage == creation_state.get('current_stage'):
                        st.write(f"🔄 {stage.title()} (진행중)")
                    else:
                        st.write(f"⏳ {stage.title()}")
        else:
            st.info("지식 창출을 시작하면 실시간 모니터링이 표시됩니다.")

    # 결과 표시 영역
    if 'creation_state' in st.session_state:
        show_creation_results()

def show_creation_process():
    """창출 프로세스 진행 상황 표시"""
    st.subheader("🔄 Multi-Agent 워크플로우")

    creation_state = st.session_state.creation_state

    # 시뮬레이션된 프로세스 실행 (실제로는 LangGraph가 처리)
    with st.container():
        # 1. Librarian (정규화)
        with st.expander("📚 Librarian - 데이터 정규화", expanded=True):
            if creation_state.get("current_stage") == "normalize":
                with st.spinner("PII 필터링 및 데이터 정규화 중..."):
                    time.sleep(1)
                    st.success("✅ 데이터 정규화 완료")
                    # 다음 단계로 이동
                    creation_state["current_stage"] = "sample"
                    creation_state["stages_completed"].append("normalize")
                    st.rerun()
            elif "normalize" in creation_state.get("stages_completed", []):
                st.success("✅ 데이터 정규화 완료")
            else:
                st.info("⏳ 대기 중")

        # 2. Sampler (다양성 샘플링)
        with st.expander("🎲 Sampler - 다양성 샘플링", expanded=creation_state.get("current_stage") == "sample"):
            if creation_state.get("current_stage") == "sample":
                with st.spinner("MMR 기반 다양성 샘플링 중..."):
                    time.sleep(1.5)
                    st.success("✅ 10개 샘플 선택 완료")
                    creation_state["current_samples"] = [f"Sample_{i+1}" for i in range(10)]
                    creation_state["current_stage"] = "summarize"
                    creation_state["stages_completed"].append("sample")
                    st.rerun()
            elif "sample" in creation_state.get("stages_completed", []):
                st.success(f"✅ {len(creation_state.get('current_samples', []))}개 샘플 선택 완료")
            else:
                st.info("⏳ 대기 중")

        # 3. Summarizer (구조화 요약)
        with st.expander("📝 Summarizer - 구조화 요약", expanded=creation_state.get("current_stage") == "summarize"):
            if creation_state.get("current_stage") == "summarize":
                with st.spinner("구조화된 요약 생성 중..."):
                    time.sleep(2)
                    st.success("✅ 구조화 요약 완료")
                    creation_state["summaries"] = [f"Summary_{i+1}" for i in range(10)]
                    creation_state["current_stage"] = "synthesize"
                    creation_state["stages_completed"].append("summarize")
                    st.rerun()
            elif "summarize" in creation_state.get("stages_completed", []):
                st.success(f"✅ {len(creation_state.get('summaries', []))}개 요약 생성 완료")
            else:
                st.info("⏳ 대기 중")

        # 4. Synthesizer (융합 제안)
        with st.expander("🧬 Synthesizer - 지식 융합", expanded=creation_state.get("current_stage") == "synthesize"):
            if creation_state.get("current_stage") == "synthesize":
                with st.spinner("아날로지 및 패턴 기반 융합 제안 생성 중..."):
                    time.sleep(2.5)
                    st.success("✅ 융합 제안 생성 완료")
                    creation_state["proposals"] = [f"Proposal_{i+1}" for i in range(5)]
                    creation_state["current_stage"] = "verify"
                    creation_state["stages_completed"].append("synthesize")
                    st.rerun()
            elif "synthesize" in creation_state.get("stages_completed", []):
                st.success(f"✅ {len(creation_state.get('proposals', []))}개 융합 제안 생성 완료")
            else:
                st.info("⏳ 대기 중")

        # 5. Verifier (검증)
        with st.expander("🔍 Verifier - 검증", expanded=creation_state.get("current_stage") == "verify"):
            if creation_state.get("current_stage") == "verify":
                with st.spinner("반례/편향/외삽 위험 검증 중..."):
                    time.sleep(2)
                    st.success("✅ 검증 완료")
                    creation_state["verdicts"] = [f"Verdict_{i+1}" for i in range(5)]
                    creation_state["current_stage"] = "productize"
                    creation_state["stages_completed"].append("verify")
                    st.rerun()
            elif "verify" in creation_state.get("stages_completed", []):
                st.success(f"✅ {len(creation_state.get('verdicts', []))}개 제안 검증 완료")
            else:
                st.info("⏳ 대기 중")

        # 6. Productizer (K-Note 생성)
        with st.expander("📋 Productizer - K-Note 생성", expanded=creation_state.get("current_stage") == "productize"):
            if creation_state.get("current_stage") == "productize":
                with st.spinner("승인된 제안을 K-Note로 변환 중..."):
                    time.sleep(1.5)
                    st.success("✅ K-Note 생성 완료")
                    creation_state["knotes"] = [f"KNote_{i+1}" for i in range(3)]
                    creation_state["current_stage"] = "score"
                    creation_state["stages_completed"].append("productize")
                    st.rerun()
            elif "productize" in creation_state.get("stages_completed", []):
                st.success(f"✅ {len(creation_state.get('knotes', []))}개 K-Note 생성 완료")
            else:
                st.info("⏳ 대기 중")

        # 7. Evaluator (평가)
        with st.expander("📊 Evaluator - 평가", expanded=creation_state.get("current_stage") == "score"):
            if creation_state.get("current_stage") == "score":
                with st.spinner("신규성, 커버리지, 유용성 평가 중..."):
                    time.sleep(1)
                    # 시뮬레이션된 점수
                    import random
                    scores = {
                        "novelty": round(random.uniform(0.6, 0.9), 2),
                        "coverage": round(random.uniform(0.6, 0.9), 2),
                        "utility": round(random.uniform(0.6, 0.9), 2)
                    }
                    creation_state["scores"] = scores
                    avg_score = sum(scores.values()) / len(scores)

                    if avg_score >= 0.75:
                        st.success(f"✅ 목표 달성! 평균 점수: {avg_score:.2f}")
                        creation_state["is_running"] = False
                    else:
                        st.warning(f"⚠️ 목표 미달. 평균 점수: {avg_score:.2f}")
                        if creation_state["iteration"] < creation_state["max_iterations"]:
                            creation_state["iteration"] += 1
                            creation_state["current_stage"] = "sample"
                            creation_state["stages_completed"] = ["normalize"]  # 일부 단계 유지
                            st.info(f"🔄 반복 {creation_state['iteration']} 시작")
                        else:
                            st.error("❌ 최대 반복 횟수 도달. 프로세스 종료")
                            creation_state["is_running"] = False

                    creation_state["stages_completed"].append("score")
                    st.rerun()
            elif "score" in creation_state.get("stages_completed", []):
                scores = creation_state.get("scores", {})
                if scores:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("신규성", f"{scores.get('novelty', 0):.2f}")
                    with col2:
                        st.metric("커버리지", f"{scores.get('coverage', 0):.2f}")
                    with col3:
                        st.metric("유용성", f"{scores.get('utility', 0):.2f}")
            else:
                st.info("⏳ 대기 중")

def show_creation_results():
    """창출 결과 표시"""
    creation_state = st.session_state.creation_state

    if not creation_state.get("is_running") and creation_state.get("knotes"):
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
        for i, knote in enumerate(creation_state.get("knotes", []), 1):
            with st.expander(f"K-Note {i}: {knote}"):
                st.write("**제목**: 새로운 지식 패턴 발견")
                st.write("**카테고리**: 패턴 융합")
                st.write("**신뢰도**: 85%")
                st.write("**근거 문서**: 3개")
                st.text_area("내용", "여기에 실제 생성된 K-Note 내용이 표시됩니다...", height=100, key=f"knote_{i}")

                col_action1, col_action2 = st.columns(2)
                with col_action1:
                    st.button(f"📚 게시판에 등록", key=f"save_knote_{i}")
                with col_action2:
                    st.button(f"💾 VectorDB에 저장", key=f"vector_knote_{i}")

# 메인 앱
def main():
    initialize_session_state()

    # 사이드바 네비게이션
    selected_page = sidebar_navigation()

    # 페이지 라우팅
    if selected_page == "챗봇":
        chatbot_page()
    elif selected_page == "지식등록":
        knowledge_registration_page()
    elif selected_page == "게시판":
        board_page()
    elif selected_page == "지식창출":
        knowledge_creation_page()

if __name__ == "__main__":
    main()