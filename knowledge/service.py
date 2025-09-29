"""
지식 관리 서비스 로직
"""
import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from utils.file_processor import FileProcessor
from services.document_analyzer import DocumentAnalyzer

class KnowledgeService:
    """지식 관리 서비스 클래스"""

    def __init__(self):
        self.file_processor = FileProcessor()
        self.document_analyzer = DocumentAnalyzer()

    def process_uploaded_file(self, uploaded_file) -> Tuple[Optional[Dict], Optional[str]]:
        """업로드된 파일 처리"""
        # 파일 유효성 검사
        is_valid, error_message = self.file_processor.validate_file(uploaded_file)

        if not is_valid:
            st.error(error_message)
            return None, None

        # 파일 정보 추출
        file_info = self.file_processor.get_file_info(uploaded_file)

        # 파일 내용 추출
        file_content = self.file_processor.extract_text(uploaded_file)

        return file_info, file_content

    def analyze_document(self, content: str, filename: str, settings: Dict = None) -> Dict:
        """문서 분석 실행"""
        # 분석 시뮬레이션 (실제 환경에서는 시간 단축)
        time.sleep(2)

        # 설정에 따른 분석 실행
        if settings and settings.get('depth') == '전문가':
            return self._expert_analysis(content, filename)
        elif settings and settings.get('depth') == '상세':
            return self._detailed_analysis(content, filename)
        else:
            return self.document_analyzer.analyze_document(content, filename)

    def _expert_analysis(self, content: str, filename: str) -> Dict:
        """전문가 수준 분석"""
        basic_result = self.document_analyzer.analyze_document(content, filename)

        # 전문가 분석 추가 요소
        expert_additions = {
            "expert_insights": [
                "업계 표준과의 비교 분석 필요",
                "최신 동향 반영 권장",
                "실무 케이스 스터디 추가 제안",
                "정량적 데이터 보강 필요"
            ],
            "technical_assessment": {
                "complexity_level": "중급",
                "target_audience": "실무진",
                "implementation_difficulty": "보통"
            },
            "quality_score": min(basic_result["quality_score"] + 10, 100)  # 전문가 분석으로 품질 점수 향상
        }

        # 기본 결과에 전문가 분석 추가
        result = {**basic_result, **expert_additions}
        return result

    def _detailed_analysis(self, content: str, filename: str) -> Dict:
        """상세 분석"""
        basic_result = self.document_analyzer.analyze_document(content, filename)

        # 상세 분석 추가 요소
        detailed_additions = {
            "detailed_metrics": {
                "readability_score": 78,
                "technical_accuracy": 85,
                "completeness_ratio": 0.75
            },
            "section_analysis": [
                {"section": "개요", "completeness": 90, "quality": 85},
                {"section": "본문", "completeness": 70, "quality": 80},
                {"section": "결론", "completeness": 60, "quality": 75}
            ],
            "quality_score": min(basic_result["quality_score"] + 5, 100)  # 상세 분석으로 품질 점수 향상
        }

        result = {**basic_result, **detailed_additions}
        return result

    def save_to_vector_db(self, analysis_result: Dict) -> Dict:
        """VectorDB에 저장"""
        try:
            vector_entry = {
                "content": analysis_result['enhanced_content'],
                "filename": st.session_state.current_file_info['name'],
                "metadata": analysis_result.get('metadata', {}),
                "quality_score": analysis_result['quality_score'],
                "timestamp": datetime.now(),
                "original_length": analysis_result['original_length'],
                "analysis_type": st.session_state.get('analysis_settings', {}).get('depth', '기본')
            }

            if 'vector_db' not in st.session_state:
                st.session_state.vector_db = []

            st.session_state.vector_db.append(vector_entry)

            return {
                "success": True,
                "message": "VectorDB 저장 완료",
                "count": len(st.session_state.vector_db)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"저장 실패: {str(e)}"
            }

    def save_to_board(self, analysis_result: Dict) -> Dict:
        """게시판에 저장"""
        try:
            board_post = {
                "title": f"[AI 보완] {st.session_state.current_file_info['name']}",
                "content": analysis_result['enhanced_content'],
                "author": "AI Knowledge System",
                "timestamp": datetime.now(),
                "views": 0,
                "quality_score": analysis_result['quality_score'],
                "file_info": st.session_state.current_file_info,
                "issues_found": analysis_result['issues_found'],
                "improvements": analysis_result['improvements'],
                "analysis_type": st.session_state.get('analysis_settings', {}).get('depth', '기본')
            }

            if 'board_posts' not in st.session_state:
                st.session_state.board_posts = []

            st.session_state.board_posts.append(board_post)

            return {
                "success": True,
                "message": "게시판 등록 완료",
                "count": len(st.session_state.board_posts)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"등록 실패: {str(e)}"
            }

    def bulk_vector_to_board(self) -> Dict:
        """VectorDB 문서들을 게시판으로 일괄 이동"""
        vector_db = st.session_state.get('vector_db', [])

        if not vector_db:
            return {
                "success": False,
                "message": "VectorDB에 문서가 없습니다",
                "count": 0
            }

        if 'board_posts' not in st.session_state:
            st.session_state.board_posts = []

        moved_count = 0
        for doc in vector_db:
            # 이미 게시판에 있는지 확인
            existing = any(
                post.get('title', '').endswith(doc.get('filename', ''))
                for post in st.session_state.board_posts
            )

            if not existing:
                board_post = {
                    "title": f"[VectorDB 이동] {doc.get('filename', 'Unknown')}",
                    "content": doc.get('content', ''),
                    "author": "Knowledge Management System",
                    "timestamp": datetime.now(),
                    "views": 0,
                    "quality_score": doc.get('quality_score', 0),
                    "file_info": {"name": doc.get('filename', 'Unknown')},
                    "issues_found": [],
                    "improvements": [],
                    "source": "vector_db"
                }

                st.session_state.board_posts.append(board_post)
                moved_count += 1

        return {
            "success": True,
            "message": f"{moved_count}개 문서 이동 완료",
            "count": moved_count
        }

    def get_knowledge_stats(self) -> Dict:
        """지식 관리 통계"""
        vector_db = st.session_state.get('vector_db', [])
        board_posts = st.session_state.get('board_posts', [])

        # 평균 품질 점수
        all_scores = []
        for doc in vector_db:
            if 'quality_score' in doc:
                all_scores.append(doc['quality_score'])
        for post in board_posts:
            if 'quality_score' in post:
                all_scores.append(post['quality_score'])

        avg_quality = sum(all_scores) / len(all_scores) if all_scores else 0

        # 오늘 등록된 문서 수
        today = datetime.now().date()
        today_count = 0

        for doc in vector_db:
            if doc.get('timestamp') and doc['timestamp'].date() == today:
                today_count += 1

        for post in board_posts:
            if post.get('timestamp') and post['timestamp'].date() == today:
                today_count += 1

        return {
            'vector_count': len(vector_db),
            'board_count': len(board_posts),
            'avg_quality': avg_quality,
            'today_count': today_count,
            'total_documents': len(vector_db) + len(board_posts)
        }

    def get_file_info(self, uploaded_file) -> Dict:
        """파일 정보 반환"""
        return self.file_processor.get_file_info(uploaded_file)

    def search_knowledge(self, query: str) -> Dict:
        """지식 검색"""
        vector_db = st.session_state.get('vector_db', [])
        board_posts = st.session_state.get('board_posts', [])

        results = {
            'vector_results': [],
            'board_results': [],
            'total_count': 0
        }

        query_lower = query.lower()

        # VectorDB 검색
        for doc in vector_db:
            if query_lower in doc.get('content', '').lower() or query_lower in doc.get('filename', '').lower():
                results['vector_results'].append(doc)

        # 게시판 검색
        for post in board_posts:
            if query_lower in post.get('content', '').lower() or query_lower in post.get('title', '').lower():
                results['board_results'].append(post)

        results['total_count'] = len(results['vector_results']) + len(results['board_results'])
        return results