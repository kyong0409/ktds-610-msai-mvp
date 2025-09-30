import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import tempfile
from typing import List, Dict
from knowledge.service import KnowledgeService, RAGService
from knowledge.creation_engine import KnowledgeCreationEngine
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
    if 'rag_service' not in st.session_state:
        st.session_state.rag_service = RAGService()
    if 'creation_engine' not in st.session_state:
        st.session_state.creation_engine = KnowledgeCreationEngine()


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
            with st.spinner("Multi-Agent 지식 창출을 시작합니다..."):
                try:
                    # LangGraph 엔진 실행
                    creation_engine = st.session_state.creation_engine
                    result = creation_engine.run(max_iter=max_iterations)

                    # 디버깅: 결과 출력
                    st.write("**디버그: 생성 결과**")
                    st.write(f"- 반복 횟수: {result.get('iter', 0)}")
                    st.write(f"- 샘플 수: {len(result.get('samples', []))}")
                    st.write(f"- 요약 수: {len(result.get('summaries', []))}")
                    st.write(f"- 제안 수: {len(result.get('proposals', []))}")
                    st.write(f"- 검증 수: {len(result.get('verdicts', []))}")
                    st.write(f"- K-Note 수: {len(result.get('knotes', []))}")
                    st.write(f"- 점수: {result.get('scores', {})}")
                    st.write(f"- 종료 이유: {result.get('stop_reason', 'unknown')}")

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
                        st.success(f"✅ 지식 창출 완료! {len(result['knotes'])}개의 K-Note 생성됨")
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
        knotes = creation_state.get("knotes", [])

        if isinstance(knotes, list) and len(knotes) > 0:
            for i, knote in enumerate(knotes, 1):
                # K-Note가 딕셔너리인지 확인
                if isinstance(knote, dict):
                    k_note_id = knote.get("k_note_id", f"K-Note {i}")
                    title = knote.get("title", "제목 없음")
                    proposal = knote.get("proposal", "")
                    evidence = knote.get("evidence", [])

                    with st.expander(f"📝 {k_note_id}: {title}"):
                        st.write(f"**제목**: {title}")
                        st.write(f"**상태**: {knote.get('status', 'draft')}")
                        st.write(f"**버전**: {knote.get('version', '1.0')}")

                        # 제안 내용
                        st.markdown("### 💡 제안 내용")
                        st.write(proposal)

                        # 적용 가능성
                        if "applicability" in knote:
                            st.markdown("### 🎯 적용 가능성")
                            applicability = knote["applicability"]
                            if "when" in applicability:
                                st.write(f"**적용 시기:** {', '.join(applicability['when'])}")
                            if "when_not" in applicability:
                                st.write(f"**적용 제외:** {', '.join(applicability['when_not'])}")

                        # 근거 문서
                        if evidence:
                            st.markdown("### 📚 근거 문서")
                            for ev in evidence[:3]:
                                doc_id = ev.get("doc_id", "unknown")
                                confidence = ev.get("confidence", 0)
                                st.write(f"- 문서: {doc_id} (신뢰도: {confidence})")

                        # 위험 및 제한사항
                        if "risks_limits" in knote and knote["risks_limits"]:
                            st.markdown("### ⚠️ 위험 및 제한사항")
                            for risk in knote["risks_limits"]:
                                st.write(f"- {risk}")

                        # 액션 버튼
                        col_action1, col_action2 = st.columns(2)
                        with col_action1:
                            if st.button(f"📚 게시판에 등록", key=f"save_knote_{i}"):
                                st.success("게시판 등록 기능은 추후 구현 예정입니다.")
                        with col_action2:
                            if st.button(f"💾 VectorDB에 저장", key=f"vector_knote_{i}"):
                                st.success("VectorDB 저장 기능은 추후 구현 예정입니다.")
                else:
                    # 문자열인 경우 (기존 시뮬레이션 형식)
                    with st.expander(f"K-Note {i}: {knote}"):
                        st.write("**제목**: 새로운 지식 패턴 발견")
                        st.text_area("내용", str(knote), height=100, key=f"knote_{i}")
        else:
            st.info("생성된 K-Note가 없습니다.")

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