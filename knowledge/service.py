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
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
import sqlite3
import uuid
from typing import List

load_dotenv()

AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

key = AzureKeyCredential(AZURE_DOCUMENT_INTELLIGENCE_KEY)

# Azure OpenAI 설정
endpoint = os.environ["AZURE_ENDPOINT"]
api_key = os.environ["AZURE_AI_FOUNDRY_KEY"]
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-12-01-preview")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

print("endpoint: ", endpoint)
print("api_key: ", api_key)
print("api_version: ", api_version)

# 배포 이름: Foundry에서 만든 배포명과 동일해야 함
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
print("DEPLOYMENT_NAME: ", DEPLOYMENT_NAME)
llm = AzureChatOpenAI(
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
        
        template = """
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
            "metadata": {{
                "type": "",
                "topic": "",
                "author": "",
                "date": "",
                "project_area": "",
                "keywords": []
            }},
            "structure": [
                {{
                "section": "",
                "summary": ""
                }}
            ],
            "usability": "이 문서가 지식자산으로서 어떻게 활용될 수 있는지 설명",
            "improvements": [
                "보완점1",
                "보완점2",
                "보완점3"
            ]
        }}
        """

        # LLM 분석 시도
        try:
            prompt = PromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()

            print("content: ", content[:50])

            response = chain.invoke({
                "content": content
            })

            # JSON 응답을 파싱하여 기존 구조와 호환되도록 변환
            import json
            # JSON 블록에서 실제 JSON만 추출
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            llm_result = json.loads(json_str)

            # 기존 DocumentAnalyzer 구조와 호환되도록 변환
            enhanced_content = self._create_enhanced_content(content, llm_result)
            return {
                "original_content": content,
                "enhanced_content": enhanced_content,
                "quality_score": self._calculate_quality_score(content, llm_result),
                "original_length": len(content),
                "enhanced_length": len(enhanced_content),
                "issues_found": llm_result.get("improvements", []),
                "improvements": llm_result.get("improvements", []),
                "metadata": {
                    "analyzed_at": datetime.now(),
                    "filename": filename,
                    "analyzer_version": "LLM-1.0",
                    "llm_metadata": llm_result.get("metadata", {})
                },
                "llm_analysis": llm_result
            }
        except Exception as e:
            # LLM 분석 실패시 기존 DocumentAnalyzer로 fallback
            print(f"LLM 분석 실패 (네트워크 오류 또는 파싱 실패), fallback 사용: {e}")
            return self._fallback_analysis(content, filename, str(e))

    def _create_enhanced_content(self, original_content: str, llm_result: Dict) -> str:
        """LLM 분석 결과를 기반으로 향상된 콘텐츠 생성"""
        metadata = llm_result.get("metadata", {})
        structure = llm_result.get("structure", [])
        usability = llm_result.get("usability", "")
        improvements = llm_result.get("improvements", [])

        enhanced = f"""# {metadata.get('topic', '문서 분석 결과')}

## 📋 문서 메타데이터
- **문서 종류**: {metadata.get('type', '미분류')}
- **주제**: {metadata.get('topic', '미확인')}
- **작성자**: {metadata.get('author', '미확인')}
- **작성일**: {metadata.get('date', '미확인')}
- **프로젝트/분야**: {metadata.get('project_area', '미확인')}
- **키워드**: {', '.join(metadata.get('keywords', []))}

## 📑 원본 내용
{original_content}

## 🔍 문서 구조 분석
"""
        for section in structure:
            enhanced += f"\n### {section.get('section', '섹션')}\n{section.get('summary', '내용 요약')}\n"

        enhanced += f"""
## 💡 활용 가능성
{usability}

## ✨ 개선 제안사항
"""
        for i, improvement in enumerate(improvements, 1):
            enhanced += f"{i}. {improvement}\n"

        enhanced += """
---
*본 문서는 AI 지식관리 시스템에 의해 분석 및 보완되었습니다.*
"""
        return enhanced.strip()

    def _calculate_quality_score(self, content: str, llm_result: Dict) -> int:
        """콘텐츠와 LLM 분석을 기반으로 품질 점수 계산"""
        base_score = 60

        # 길이 기반 점수
        length_score = min(len(content) // 100, 20)

        # 메타데이터 완성도
        metadata = llm_result.get("metadata", {})
        metadata_score = 0
        for key in ["type", "topic", "author", "date", "project_area"]:
            if metadata.get(key) and metadata[key] != "미확인" and metadata[key] != "":
                metadata_score += 2

        # 키워드 개수
        keywords_score = min(len(metadata.get("keywords", [])) * 1, 10)

        # 구조 분석 품질
        structure_score = min(len(llm_result.get("structure", [])) * 3, 15)

        total_score = base_score + length_score + metadata_score + keywords_score + structure_score
        return min(total_score, 100)

    def _fallback_analysis(self, content: str, filename: str, error_msg: str) -> Dict:
        """LLM 분석 실패시 fallback 분석"""
        basic_result = self.document_analyzer.analyze_document(content, filename)

        # 에러 정보를 메타데이터에 추가
        basic_result["metadata"]["error_info"] = {
            "llm_error": error_msg,
            "fallback_used": True,
            "error_time": datetime.now()
        }
        basic_result["metadata"]["analyzer_version"] = "Fallback-1.0"

        # 에러 관련 개선사항 추가
        if "Connection error" in error_msg or "getaddrinfo failed" in error_msg:
            basic_result["issues_found"].insert(0, "⚠️ AI 분석 서비스 연결 실패 - 네트워크 연결을 확인해주세요")
            basic_result["improvements"].insert(0, "네트워크 연결 상태 확인 후 재시도")

        return basic_result

    def generate_enhanced_knowledge_document(self, analysis_result: Dict, filename: str) -> Dict:
        """분석 결과를 바탕으로 보완된 지식 문서 생성"""
        
        template = """
        [역할]  
        당신은 IT 회사의 지식 관리 전문가이자 기술 문서 편집자입니다.  
        당신의 임무는 입력된 문서와 보완사항을 종합하여, 사내에서 활용 가능한 "표준 지식 문서" 형태로 재구성하는 것입니다.  

        [입력]  
        1. 원본 문서 (Original Document)  
        {original_content}
        2. 보완사항 (Improvement Points)  
        {improvement_points}

        [지시사항]  
        1. 원본 문서를 분석하여 **핵심 내용**을 유지합니다.  
        2. 보완사항을 반영하여 내용의 공백을 채우거나 표현을 개선합니다.  
        3. 최종 산출물은 다음 표준 형식을 따릅니다:  

        ### 메타데이터
        - 문서 제목: # 문서의 주제/핵심 내용이 잘 나타나도록 작성합니다.  
        - 작성자: 원문에서 발견되면 추출, 없으면 "미확인"
        - 작성일:  
        - 버전:  
        - 프로젝트/적용 분야: 문맥에서 유추
        - 주요 태그(키워드):  

        ### 문서 본문
        1. 목적 (Purpose)  
        2. 배경 및 문제 정의 (Background / Problem Statement)  
        3. 접근 방법 및 절차 (Approach / Methodology)  
        4. 결과 및 성과 (Results / Outcomes)  
        5. 한계 및 교훈 (Limitations / Lessons Learned)  
        6. 적용 및 재사용 방안 (Application / Reusability)  
        7. 참고 자료 (References)  

        4. 내용이 부족한 경우, 보완사항을 활용하여 합리적으로 보강합니다. 단, 근거 없이 새로운 사실을 창작하지는 않습니다.  
        5. 표현은 **명확하고 간결하며, 사내 검색/학습에 적합한 용어**를 사용합니다.  

        [출력]  
        위 형식에 맞춘 **완성된 지식 문서**를 제공합니다.  
        """

        prompt = PromptTemplate.from_template(template)

        chain = prompt | llm | StrOutputParser()

        # LLM으로 지식 문서 생성 시도
        try:
            response = chain.invoke({
                "original_content": analysis_result['original_content'],
                "improvement_points": analysis_result['improvements']
            })

            # 생성된 문서를 딕셔너리 형태로 반환
            enhanced_document = {
                "enhanced_content": response,
                "quality_score": analysis_result.get('quality_score', 70),
                "generation_metadata": {
                    "generated_at": datetime.now(),
                    "filename": filename,
                    "method": "LLM Enhanced Generation",
                    "version": "1.0"
                }
            }

            return enhanced_document

        except Exception as e:
            # LLM 생성 실패시 fallback
            print(f"LLM 지식 문서 생성 실패, fallback 사용: {e}")

            # 기본 템플릿으로 fallback
            fallback_content = f"""# {filename.split('.')[0]} - 지식 문서

                                    ## 📋 메타데이터
                                    - **문서 제목**: {filename}
                                    - **작성자**: 미확인
                                    - **작성일**: {datetime.now().strftime('%Y-%m-%d')}
                                    - **버전**: 1.0
                                    - **프로젝트/적용 분야**: 미확인
                                    - **주요 태그**: 분석 중

                                    ## 📖 문서 본문

                                    ### 1. 목적 (Purpose)
                                    이 문서는 지식 관리 시스템을 통해 분석된 내용을 정리한 문서입니다.

                                    ### 2. 원본 내용
                                    {analysis_result.get('original_content', '내용 없음')}

                                    ### 3. 발견된 개선사항
                                    """
            for i, improvement in enumerate(analysis_result.get('improvements', []), 1):
                fallback_content += f"{i}. {improvement}\n"

            fallback_content += """
                                ### 4. 적용 및 재사용 방안
                                - 향후 유사한 프로젝트에서 참고 자료로 활용 가능
                                - 개선사항을 반영하여 문서 품질 향상 필요

                                ---
                                *본 문서는 AI 지식관리 시스템에 의해 생성되었습니다.*
                                """

            enhanced_document = {
                "enhanced_content": fallback_content,
                "quality_score": analysis_result.get('quality_score', 70),
                "generation_metadata": {
                    "generated_at": datetime.now(),
                    "filename": filename,
                    "method": "Fallback Template Generation",
                    "version": "1.0",
                    "error": str(e)
                }
            }

            return enhanced_document

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


class RAGService:
    """RAG를 위한 텍스트 분할 및 벡터 DB 관리 클래스"""

    def __init__(self, chroma_persist_directory: str = "./data/chroma_db",
                 sqlite_db_path: str = "./data/board.db"):
        """
        RAG 서비스 초기화

        Args:
            chroma_persist_directory: ChromaDB 저장 경로
            sqlite_db_path: SQLite 데이터베이스 경로
        """
        # Azure OpenAI Embeddings 초기화
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=embedding_api_version,
            azure_deployment=embedding_deployment
        )

        # ChromaDB 초기화
        self.chroma_persist_directory = chroma_persist_directory
        os.makedirs(chroma_persist_directory, exist_ok=True)

        # SQLite DB 초기화
        self.sqlite_db_path = sqlite_db_path
        os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
        self._initialize_board_db()

    def _initialize_board_db(self):
        """게시판 DB 테이블 초기화"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS board_posts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                enhanced_doc_url TEXT,
                original_doc_url TEXT,
                author TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                views INTEGER DEFAULT 0,
                quality_score INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

    def split_text(self, text: str, split_type: str = "semantic",
                   chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트
            split_type: 분할 방식 ("character", "recursive", "semantic", "token")
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침 크기

        Returns:
            분할된 텍스트 청크 리스트
        """
        try:
            if split_type == "character":
                splitter = CharacterTextSplitter(
                    separator="\n\n",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            elif split_type == "recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", " ", ""]
                )
            elif split_type == "semantic":
                splitter = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type="gradient"
                )
            elif split_type == "token":
                splitter = TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                raise ValueError(f"지원하지 않는 분할 방식: {split_type}")

            chunks = splitter.split_text(text)
            return chunks

        except Exception as e:
            print(f"텍스트 분할 중 오류 발생: {e}")
            # Fallback: recursive splitter 사용
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return splitter.split_text(text)

    def embed_and_store(self, text: str, metadata: Dict = None,
                       split_type: str = "semantic",
                       chunk_size: int = 1000,
                       chunk_overlap: int = 200,
                       collection_name: str = "knowledge_base") -> Dict:
        """
        텍스트를 청크로 분할하고 벡터 DB에 임베딩하여 저장

        Args:
            text: 저장할 텍스트
            metadata: 메타데이터
            split_type: 분할 방식
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            collection_name: ChromaDB 컬렉션 이름

        Returns:
            저장 결과 딕셔너리
        """
        try:
            # 텍스트 분할
            chunks = self.split_text(text, split_type, chunk_size, chunk_overlap)

            if not chunks:
                return {
                    "success": False,
                    "message": "텍스트 분할 실패",
                    "chunk_count": 0
                }

            # 메타데이터 설정
            if metadata is None:
                metadata = {}

            # 각 청크에 고유 ID 부여
            doc_id = metadata.get('doc_id', str(uuid.uuid4()))
            metadata['doc_id'] = doc_id

            # ChromaDB에 저장
            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                metadatas=[{**metadata, 'chunk_id': i} for i in range(len(chunks))],
                collection_name=collection_name,
                persist_directory=self.chroma_persist_directory
            )

            return {
                "success": True,
                "message": "VectorDB 저장 완료",
                "chunk_count": len(chunks),
                "doc_id": doc_id,
                "collection_name": collection_name
            }

        except Exception as e:
            print(f"VectorDB 저장 중 오류 발생: {e}")
            return {
                "success": False,
                "message": f"저장 실패: {str(e)}",
                "chunk_count": 0
            }

    def search_similar(self, query: str, k: int = 5,
                      collection_name: str = "knowledge_base") -> List[Dict]:
        """
        유사도 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            collection_name: 검색할 컬렉션 이름

        Returns:
            검색 결과 리스트
        """
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )

            results = vectorstore.similarity_search_with_score(query, k=k)

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })

            return formatted_results

        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return []

    def save_to_board_db(self, title: str, content: str,
                        enhanced_doc_url: str = None,
                        original_doc_url: str = None,
                        author: str = "AI Knowledge System",
                        quality_score: int = 0,
                        metadata: Dict = None) -> Dict:
        """
        게시판 DB에 문서 저장

        Args:
            title: 문서 제목
            content: 문서 내용 (보완된 내용)
            enhanced_doc_url: 보완된 문서 다운로드 링크
            original_doc_url: 원본 문서 다운로드 링크
            author: 작성자
            quality_score: 품질 점수
            metadata: 추가 메타데이터 (JSON 형식으로 저장)

        Returns:
            저장 결과 딕셔너리
        """
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()

            post_id = str(uuid.uuid4())
            import json

            # datetime 객체를 문자열로 변환
            if metadata:
                metadata_copy = {}
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        # 중첩된 딕셔너리 처리
                        metadata_copy[key] = {}
                        for k, v in value.items():
                            if isinstance(v, datetime):
                                metadata_copy[key][k] = v.isoformat()
                            else:
                                metadata_copy[key][k] = v
                    elif isinstance(value, datetime):
                        metadata_copy[key] = value.isoformat()
                    else:
                        metadata_copy[key] = value
                metadata_json = json.dumps(metadata_copy, ensure_ascii=False)
            else:
                metadata_json = None

            cursor.execute("""
                INSERT INTO board_posts
                (id, title, content, enhanced_doc_url, original_doc_url,
                 author, quality_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (post_id, title, content, enhanced_doc_url, original_doc_url,
                  author, quality_score, metadata_json))

            conn.commit()
            conn.close()

            return {
                "success": True,
                "message": "게시판 저장 완료",
                "post_id": post_id
            }

        except Exception as e:
            print(f"게시판 저장 중 오류 발생: {e}")
            return {
                "success": False,
                "message": f"저장 실패: {str(e)}"
            }

    def get_board_posts(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        게시판 게시글 조회

        Args:
            limit: 조회할 게시글 수
            offset: 시작 위치

        Returns:
            게시글 리스트
        """
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM board_posts
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))

            rows = cursor.fetchall()
            conn.close()

            import json
            posts = []
            for row in rows:
                post = dict(row)
                if post.get('metadata'):
                    post['metadata'] = json.loads(post['metadata'])
                posts.append(post)

            return posts

        except Exception as e:
            print(f"게시글 조회 중 오류 발생: {e}")
            return []

    def get_board_post_by_id(self, post_id: str) -> Optional[Dict]:
        """
        특정 게시글 조회 및 조회수 증가

        Args:
            post_id: 게시글 ID

        Returns:
            게시글 정보
        """
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 조회수 증가
            cursor.execute("""
                UPDATE board_posts
                SET views = views + 1
                WHERE id = ?
            """, (post_id,))

            # 게시글 조회
            cursor.execute("""
                SELECT * FROM board_posts WHERE id = ?
            """, (post_id,))

            row = cursor.fetchone()
            conn.commit()
            conn.close()

            if row:
                import json
                post = dict(row)
                if post.get('metadata'):
                    post['metadata'] = json.loads(post['metadata'])
                return post

            return None

        except Exception as e:
            print(f"게시글 조회 중 오류 발생: {e}")
            return None

    def delete_board_post(self, post_id: str) -> Dict:
        """
        게시글 삭제

        Args:
            post_id: 게시글 ID

        Returns:
            삭제 결과 딕셔너리
        """
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM board_posts WHERE id = ?", (post_id,))

            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()

            if deleted:
                return {
                    "success": True,
                    "message": "게시글 삭제 완료"
                }
            else:
                return {
                    "success": False,
                    "message": "게시글을 찾을 수 없습니다"
                }

        except Exception as e:
            print(f"게시글 삭제 중 오류 발생: {e}")
            return {
                "success": False,
                "message": f"삭제 실패: {str(e)}"
            }