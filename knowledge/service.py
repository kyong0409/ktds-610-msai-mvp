"""
지식 관리 서비스 로직
"""
import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from utils.file_processor import FileProcessor
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
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
import sqlite3
import uuid
from typing import List

from knowledge.prompts import (
    ANALYZE_DOCUMENT_PROMPT,
    GENERATE_ENHANCED_KNOWLEDGE_DOCUMENT_PROMPT,
    KNOTE_TO_STANDARD_DOC_PROMPT,
)

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

        # MarkItDown 초기화 (Azure Document Intelligence 선택적 사용)
        try:
            if AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY:
                print(f"[INFO] MarkItDown with Azure Document Intelligence 초기화 중...")
                self.markitdown = MarkItDown(
                    docintel_endpoint=AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
                    docintel_credential=key
                )
                print(f"[INFO] MarkItDown 초기화 성공 (Azure DI 활성화)")
            else:
                print(f"[WARNING] Azure Document Intelligence 설정 누락, 기본 MarkItDown 사용")
                self.markitdown = MarkItDown()
                print(f"[INFO] MarkItDown 초기화 성공 (기본 모드)")
        except Exception as e:
            print(f"[ERROR] MarkItDown 초기화 실패: {e}")
            print(f"[INFO] Fallback: 기본 MarkItDown 사용")
            self.markitdown = MarkItDown()

    def convert_file_to_text(self, uploaded_file) -> Tuple[bool, str]:
        """MarkItDown을 사용하여 파일을 텍스트로 변환"""
        try:
            # 임시 파일로 저장
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            print(f"[DEBUG] MarkItDown 변환 시도: {tmp_file_path}")

            # MarkItDown으로 변환
            result = self.markitdown.convert(tmp_file_path, keep_data_uris=True)

            print(f"[DEBUG] MarkItDown 변환 성공")

            # 임시 파일 삭제
            os.unlink(tmp_file_path)

            # 업로드 파일 포인터 리셋
            uploaded_file.seek(0)

            return True, result.text_content

        except Exception as e:
            print(f"[ERROR] MarkItDown 변환 실패: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

            # 업로드 파일 포인터 리셋
            uploaded_file.seek(0)

            # Fallback: 기존 파일 처리 방식 사용
            print(f"[INFO] Fallback 텍스트 추출 시도")
            fallback_content = self._fallback_text_extraction(uploaded_file)
            if fallback_content:
                print(f"[INFO] Fallback 텍스트 추출 성공 ({len(fallback_content)} chars)")
                return True, fallback_content

            return False, f"파일 변환 중 오류 발생: {str(e)}"

    def _fallback_text_extraction(self, uploaded_file) -> str:
        """Fallback: 기본 텍스트 추출 방식"""
        try:
            # 파일 타입별 기본 텍스트 추출
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'txt':
                # 텍스트 파일은 바로 읽기
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                return content
            elif file_extension == 'pdf':
                # PDF는 기본 메시지 반환
                return f"PDF 파일: {uploaded_file.name}\n(MarkItDown 변환 실패, 수동 처리 필요)"
            elif file_extension in ['docx', 'doc']:
                # Word 파일은 기본 메시지 반환
                return f"Word 문서: {uploaded_file.name}\n(MarkItDown 변환 실패, 수동 처리 필요)"
            elif file_extension in ['pptx', 'ppt']:
                # PowerPoint 파일은 기본 메시지 반환
                return f"PowerPoint 문서: {uploaded_file.name}\n(MarkItDown 변환 실패, 수동 처리 필요)"
            else:
                return f"알 수 없는 파일 형식: {uploaded_file.name}"
        except Exception as e:
            return f"Fallback 텍스트 추출 실패: {str(e)}"

    def analyze_document(self, content: str, filename: str, settings: Dict = None) -> Dict:
        """문서 분석 실행"""
        
        template = ANALYZE_DOCUMENT_PROMPT

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
        
        template = GENERATE_ENHANCED_KNOWLEDGE_DOCUMENT_PROMPT

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

    def enhance_knote_to_standard_document(self, knote_json: Dict, additional_improvement_points: str = "") -> Dict:
        """K-Note를 표준 지식 문서로 변환"""
        
        # 입력 검증
        if not isinstance(knote_json, dict):
            raise ValueError(f"knote_json must be a dictionary, got {type(knote_json)}")
        
        template = KNOTE_TO_STANDARD_DOC_PROMPT
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        try:
            # K-Note JSON을 문자열로 변환
            import json
            
            def json_serializer(obj):
                """JSON 직렬화를 위한 커스텀 serializer"""
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            knote_json_str = json.dumps(knote_json, ensure_ascii=False, indent=2, default=json_serializer)
            
            response = chain.invoke({
                "k_note_json": knote_json_str,
                "additional_improvement_points": additional_improvement_points or ""
            })
            
            # 생성된 표준 문서를 딕셔너리 형태로 반환
            enhanced_document = {
                "enhanced_content": response,
                "quality_score": self._calculate_knote_quality_score(knote_json),
                "generation_metadata": {
                    "generated_at": datetime.now(),
                    "method": "K-Note to Standard Document Enhancement",
                    "version": "1.0",
                    "k_note_id": knote_json.get("k_note_id", "unknown")
                },
                "original_knote": knote_json
            }
            
            return enhanced_document
            
        except Exception as e:
            print(f"K-Note 표준 문서 변환 실패: {e}")
            
            # Fallback: 기본 템플릿으로 변환
            fallback_content = self._create_fallback_standard_document(knote_json)
            
            enhanced_document = {
                "enhanced_content": fallback_content,
                "quality_score": self._calculate_knote_quality_score(knote_json),
                "generation_metadata": {
                    "generated_at": datetime.now(),
                    "method": "Fallback Standard Document Generation",
                    "version": "1.0",
                    "error": str(e),
                    "k_note_id": knote_json.get("k_note_id", "unknown")
                },
                "original_knote": knote_json
            }
            
            return enhanced_document

    def _calculate_knote_quality_score(self, knote_json: Dict) -> int:
        """K-Note의 품질 점수 계산"""
        base_score = 70
        
        # 필수 필드 존재 여부
        required_fields = ["title", "proposal", "applicability", "evidence"]
        field_score = sum(10 for field in required_fields if knote_json.get(field))
        
        # Evidence 품질 점수
        evidence_score = 0
        evidence_list = knote_json.get("evidence", [])
        if evidence_list:
            # confidence 점수 평균
            confidences = [e.get("confidence", 0) for e in evidence_list if isinstance(e, dict)]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                evidence_score = int(avg_confidence * 20)  # 0-1 범위를 0-20으로 변환
        
        # 추천 실험 존재 여부
        experiment_score = 5 if knote_json.get("recommended_experiments") else 0
        
        total_score = base_score + field_score + evidence_score + experiment_score
        return min(total_score, 100)

    def _create_fallback_standard_document(self, knote_json: Dict) -> str:
        """K-Note를 기본 템플릿으로 변환 (fallback)"""
        title = knote_json.get("title", "지식 문서")
        proposal = knote_json.get("proposal", "내용 없음")
        owners = knote_json.get("owners", ["미확인"])
        version = knote_json.get("version", "1.0")
        
        # 키워드 추출
        keywords = []
        if knote_json.get("title"):
            keywords.extend(knote_json["title"].split())
        if knote_json.get("applicability", {}).get("when"):
            keywords.extend([item[:20] for item in knote_json["applicability"]["when"][:3]])
        
        fallback_content = f"""### 메타데이터
- 문서 제목: {title}
- 작성자: {', '.join(owners) if isinstance(owners, list) else str(owners)}
- 작성일: 미확인
- 버전: {version}
- 프로젝트/적용 분야: {knote_json.get('applicability', {}).get('when', ['미확인'])[0] if knote_json.get('applicability', {}).get('when') else '미확인'}
- 주요 태그(키워드): {', '.join(keywords[:8]) if keywords else '미확인'}

### 문서 본문

1. 목적 (Purpose)
{proposal}

2. 배경 및 문제 정의 (Background / Problem Statement)
"""
        
        # 적용성 정보 추가
        applicability = knote_json.get("applicability", {})
        if applicability.get("when"):
            fallback_content += f"적용 조건:\n"
            for condition in applicability["when"][:3]:
                fallback_content += f"- {condition}\n"
        
        if applicability.get("when_not"):
            fallback_content += f"\n비적용 조건:\n"
            for condition in applicability["when_not"][:3]:
                fallback_content += f"- {condition}\n"

        fallback_content += f"""
3. 접근 방법 및 절차 (Approach / Methodology)
{knote_json.get('proposal', '상세 방법론은 원본 K-Note를 참조하세요.')}

4. 결과 및 성과 (Results / Outcomes)
"""
        
        # 메트릭 효과 추가
        metrics_effect = knote_json.get("metrics_effect", {})
        if metrics_effect:
            if isinstance(metrics_effect, dict):
                for key, value in metrics_effect.items():
                    fallback_content += f"- {key}: {value}\n"
            elif isinstance(metrics_effect, list):
                for item in metrics_effect:
                    fallback_content += f"- {str(item)}\n"
            else:
                fallback_content += f"- {str(metrics_effect)}\n"
        else:
            fallback_content += "성과 지표는 원본 K-Note를 참조하세요.\n"

        fallback_content += f"""
5. 한계 및 교훈 (Limitations / Lessons Learned)
"""
        
        # 위험 및 한계 추가
        risks_limits = knote_json.get("risks_limits", [])
        if risks_limits:
            # 안전한 리스트 처리
            if isinstance(risks_limits, list):
                for risk in risks_limits[:5]:
                    fallback_content += f"- {str(risk)}\n"
            else:
                fallback_content += f"- {str(risks_limits)}\n"
        else:
            fallback_content += "한계사항은 원본 K-Note를 참조하세요.\n"

        fallback_content += f"""
6. 적용 및 재사용 방안 (Application / Reusability)
"""
        
        # 추천 실험 추가
        experiments = knote_json.get("recommended_experiments", [])
        if experiments:
            fallback_content += "권장 실험/적용 절차:\n"
            # 안전한 리스트 처리
            if isinstance(experiments, list):
                for exp in experiments[:3]:
                    if isinstance(exp, dict):
                        fallback_content += f"- {exp.get('description', str(exp))}\n"
                    else:
                        fallback_content += f"- {str(exp)}\n"
            else:
                fallback_content += f"- {str(experiments)}\n"
        else:
            fallback_content += "재사용 방안은 원본 K-Note를 참조하세요.\n"

        fallback_content += f"""
7. 참고 자료 (References)
"""
        
        # Evidence 추가
        evidence_list = knote_json.get("evidence", [])
        if evidence_list:
            # 안전한 리스트 처리
            if isinstance(evidence_list, list):
                for i, evidence in enumerate(evidence_list[:5], 1):
                    if isinstance(evidence, dict):
                        doc_id = evidence.get("doc_id", "unknown")
                        chunk_id = evidence.get("chunk_id", "unknown")
                        quote = evidence.get("quote", "내용 없음")
                        confidence = evidence.get("confidence", 0)
                        fallback_content += f"{i}. [{doc_id}#{chunk_id}] \"{quote[:100]}...\" (confidence: {confidence:.2f})\n"
                    else:
                        fallback_content += f"{i}. {str(evidence)}\n"
            else:
                fallback_content += f"1. {str(evidence_list)}\n"
        else:
            fallback_content += "추가 출처 필요\n"

        fallback_content += f"""
### 보완이 필요한 점 (Improvement Points)
- 카테고리: 문서화 / 이슈: 자동 변환으로 인한 정보 손실 / 제안: 원본 K-Note와 대조하여 내용 보완 / 근거: Fallback 템플릿 사용

---
*본 문서는 K-Note에서 자동 변환되었습니다. 원본 K-Note ID: {knote_json.get('k_note_id', 'unknown')}*
"""
        
        return fallback_content

    def upload_enhanced_document_to_blob(self, enhanced_content: str, k_note_id: str, title: str) -> str:
        """구체화된 문서를 Azure Blob Storage enhanced 컨테이너에 업로드"""
        try:
            import io
            from datetime import datetime
            
            # 파일명 생성 (K-Note ID와 제목 기반)
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{k_note_id}_{safe_title}_{timestamp}_enhanced.md"
            
            # 문서 내용을 파일 형태로 변환
            enhanced_file = io.BytesIO(enhanced_content.encode('utf-8'))
            enhanced_file.name = filename
            
            # FileProcessor를 사용하여 업로드
            enhanced_url = self.file_processor.upload_file(enhanced_file, "enhanced")
            
            return enhanced_url
            
        except Exception as e:
            print(f"Enhanced 문서 업로드 실패: {e}")
            return None

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

            # ChromaDB 호환 메타데이터로 정제 (None 값 제거, 복잡한 타입 제거)
            clean_metadata = self._clean_metadata_for_chroma(metadata)

            # ChromaDB에 저장
            vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                metadatas=[{**clean_metadata, 'chunk_id': i} for i in range(len(chunks))],
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

    def _clean_metadata_for_chroma(self, metadata: Dict) -> Dict:
        """
        ChromaDB 호환 메타데이터로 정제
        - None 값 제거
        - str, int, float, bool만 허용
        - 중첩 딕셔너리나 리스트는 JSON 문자열로 변환
        """
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue  # None 값은 제외
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (dict, list)):
                # 복잡한 타입은 JSON 문자열로 변환
                try:
                    clean[key] = json.dumps(value, ensure_ascii=False)
                except:
                    clean[key] = str(value)
            else:
                # 기타 타입은 문자열로 변환
                clean[key] = str(value)
        return clean

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