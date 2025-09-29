import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import tempfile
from typing import List, Dict
from knowledge.service import KnowledgeService

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
        "게시판": "📋"
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

# 파일 분석 시뮬레이션
def analyze_document(file_content: str, filename: str) -> Dict:
    """문서 내용을 분석하고 보완사항 제안"""
    # 시뮬레이션된 분석 결과
    analysis_result = {
        "original_length": len(file_content),
        "issues_found": [
            "목차가 누락되어 있습니다",
            "결론 부분이 불완전합니다",
            "참고문헌이 부족합니다",
            "예시가 더 필요합니다"
        ],
        "improvements": [
            "명확한 목차 구조 추가",
            "상세한 결론 및 요약 보완",
            "관련 참고문헌 추가",
            "실무 예시 및 사례 보완"
        ],
        "quality_score": 75,
        "enhanced_content": f"""
# {filename.split('.')[0]} - 보완된 버전

## 목차
1. 개요
2. 주요 내용
3. 상세 분석
4. 결론 및 요약
5. 참고문헌

## 원본 내용
{file_content}

## 보완된 내용

### 상세 분석
본 문서의 주요 내용을 바탕으로 다음과 같은 분석을 제공합니다:
- 핵심 개념 정리
- 실무 적용 방안
- 주의사항 및 제한점

### 결론 및 요약
이 문서를 통해 다음과 같은 인사이트를 얻을 수 있습니다:
1. 주요 핵심 포인트
2. 실무 적용 가능성
3. 향후 발전 방향

### 참고문헌
- 관련 업계 보고서
- 학술 논문 참조
- 실무 가이드라인
"""
    }
    return analysis_result

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

                        analysis_result = analyze_document(st.session_state.current_file_content, uploaded_file.name)
                        st.session_state.current_analysis = analysis_result
                        st.success("분석 완료!")
                else:
                    st.error("파일 내용을 먼저 변환해주세요.")

    with col2:
        st.subheader("📊 분석 결과")

        if 'current_analysis' in st.session_state:
            result = st.session_state.current_analysis

            # 품질 점수
            st.metric("품질 점수", f"{result['quality_score']}점", delta="25점 개선 가능")

            # 발견된 문제점
            st.subheader("🔍 발견된 문제점")
            for issue in result['issues_found']:
                st.warning(f"• {issue}")

            # 개선 사항
            st.subheader("✨ 제안 개선사항")
            for improvement in result['improvements']:
                st.info(f"• {improvement}")

            # 처리 버튼들
            col2_1, col2_2 = st.columns(2)

            with col2_1:
                if st.button("💾 VectorDB 저장", type="primary"):
                    with st.spinner("VectorDB에 임베딩 중..."):
                        time.sleep(1)
                        st.session_state.vector_db.append({
                            "content": result['enhanced_content'],
                            "filename": uploaded_file.name,
                            "timestamp": datetime.now()
                        })
                        st.success("VectorDB 저장 완료!")

            with col2_2:
                if st.button("📋 게시판 등록", type="secondary"):
                    with st.spinner("게시판에 등록 중..."):
                        time.sleep(1)
                        st.session_state.board_posts.append({
                            "title": f"[보완됨] {uploaded_file.name}",
                            "content": result['enhanced_content'],
                            "author": "AI System",
                            "timestamp": datetime.now(),
                            "views": 0,
                            "quality_score": result['quality_score']
                        })
                        st.success("게시판 등록 완료!")

            # 보완된 내용 미리보기
            with st.expander("📝 보완된 내용 미리보기"):
                st.text_area("", result['enhanced_content'], height=300)

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

if __name__ == "__main__":
    main()