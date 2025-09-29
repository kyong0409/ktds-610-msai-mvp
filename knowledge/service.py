"""
지식 관리 서비스 로직
"""
import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from utils.file_processor import FileProcessor
from services.document_analyzer import DocumentAnalyzer
from markitdown import MarkItDown
from langchain_openai import AzureOpenAI
from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser

load_dotenv()

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

key = AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)

# Azure OpenAI 설정
endpoint = os.environ["AZURE_ENDPOINT"]
api_key = os.environ["AZURE_AI_FOUNDRY_KEY"]
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

print("endpoint: ", endpoint)
print("api_key: ", api_key)
print("api_version: ", api_version)

# 배포 이름: Foundry에서 만든 배포명과 동일해야 함
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
print("DEPLOYMENT_NAME: ", DEPLOYMENT_NAME)
llm = AzureOpenAI(
		azure_endpoint=endpoint,
		api_key=api_key,
		api_version=api_version,
		azure_deployment=DEPLOYMENT_NAME,
		temperature=0.4,
		max_retries=3,
		timeout=30,
)

class KnowledgeService:
    """지식 관리 서비스 클래스"""

    def __init__(self):
        self.file_processor = FileProcessor()
        self.document_analyzer = DocumentAnalyzer()
        self.markitdown = MarkItDown(docintel_endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, docintel_credential=key)

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

    def convert_file_to_text(self, uploaded_file) -> Tuple[bool, str]:
        """MarkItDown을 사용하여 파일을 텍스트로 변환"""
        try:
            # 임시 파일로 저장
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # MarkItDown으로 변환
            result = self.markitdown.convert(tmp_file_path)

            # 임시 파일 삭제
            os.unlink(tmp_file_path)

            # 업로드 파일 포인터 리셋
            uploaded_file.seek(0)

            return True, result.text_content

        except Exception as e:
            # 업로드 파일 포인터 리셋
            uploaded_file.seek(0)

            # Fallback: 기존 파일 처리 방식 사용
            fallback_content = self._fallback_text_extraction(uploaded_file)
            if fallback_content:
                return True, fallback_content

            return False, f"파일 변환 중 오류 발생: {str(e)}"

    def _fallback_text_extraction(self, uploaded_file) -> str:
        """Fallback 텍스트 추출 (기존 방식)"""
        try:
            # 파일 포인터 리셋
            uploaded_file.seek(0)

            file_type = uploaded_file.type

            if file_type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            elif file_type == "application/pdf":
                try:
                    import PyPDF2
                    import io

                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
                except:
                    return f"[PDF 파일: {uploaded_file.name}] - PDF 변환을 위해 'pip install markitdown[pdf]'를 실행해주세요."
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                try:
                    from docx import Document
                    import io

                    doc = Document(io.BytesIO(uploaded_file.read()))
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except:
                    return f"[Word 파일: {uploaded_file.name}] - 파일 내용을 읽을 수 없습니다."
            else:
                return f"[{uploaded_file.name}] - 지원되지 않는 파일 형식입니다."

        except Exception as e:
            return f"[파일 처리 오류] {str(e)}"
        finally:
            # 파일 포인터 리셋
            uploaded_file.seek(0)

    def get_file_preview(self, uploaded_file) -> str:
        """파일 미리보기 텍스트 반환"""
        success, content = self.convert_file_to_text(uploaded_file)

        if success:
            # 미리보기는 처음 1000자만 표시
            preview_text = content[:1000]
            if len(content) > 1000:
                preview_text += "\n\n... (계속)"
            return preview_text
        else:
            return content  # 오류 메시지 반환

    def analyze_document(self, content: str, filename: str, settings: Dict = None) -> Dict:
        """문서 분석 실행"""
        
        template = f"""
        [역할]  
        당신은 IT 회사의 지식 관리 전문가입니다.  
        입력된 텍스트 문서(원본 지식 문서)를 분석하여, 지식자산화에 필요한 메타데이터를 추출하고 개선이 필요한 보완점을 도출하세요.  

        [입력]  
        {content}

        [지시사항]  
        다음 항목을 반드시 포함하여 분석 결과를 작성하세요.  

        ## 1. 메타데이터 추출
        - 문서 종류: {{PoC 보고서 | Lessons Learned | 기술자료 | 프로젝트 산출물 | 기타}}
        - 주제(Topic): 한 줄 요약  
        - 작성일/작성자: 원문에서 발견되면 추출, 없으면 "미확인"  
        - 프로젝트/적용 분야: 문맥에서 유추  
        - 주요 키워드(태그): 핵심 기술, 도메인, 관련 용어를 5~10개  

        ## 2. 문서 구조/목차 분석
        - 문서 내 존재하는 주요 섹션/항목 목록화  
        - 각 섹션이 다루는 내용 요약  

        ## 3. 활용 가능성 분석
        - 이 문서가 지식자산으로서 어떤 가치를 가질 수 있는지  
        - 재사용/참조 가능한 부분  

        ## 4. 보완이 필요한 점
        - 빠진 항목 (예: 목적, 결과, 교훈, 적용 방안 등)  
        - 불명확하거나 정리되지 않은 부분  
        - 검색/재사용 관점에서 개선해야 할 점  

        [출력 형식]  
        아래 JSON 구조로 결과를 제공합니다.  

        ```json
        {{
            "metadata": {
                "type": "",
                "topic": "",
                "author": "",
                "date": "",
                "project_area": "",
                "keywords": []
            },
            "structure": [
                {
                "section": "",
                "summary": ""
                }
            ],
            "usability": "이 문서가 지식자산으로서 어떻게 활용될 수 있는지 설명",
            "improvements": [
                "보완점1",
                "보완점2",
                "보완점3"
            ]
        }}
        """

        prompt = PromptTemplate.from_template(template)

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            content: content
        })

        return response

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