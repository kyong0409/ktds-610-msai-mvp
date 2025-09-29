"""
챗봇 서비스 로직
"""
import streamlit as st
from typing import List, Dict
from config.settings import get_config

class ChatbotService:
    """챗봇 서비스 클래스"""

    def __init__(self):
        self.chat_config = get_config("chat")
        self.settings = st.session_state.get('chat_settings', {
            'temperature': 0.7,
            'max_tokens': 1000
        })

    def generate_response(self, query: str) -> str:
        """사용자 질문에 대한 응답 생성"""
        # 벡터 DB에서 관련 문서 검색
        vector_db_data = st.session_state.get('vector_db', [])

        if not vector_db_data:
            return self._get_no_knowledge_response()

        # 관련 문서 검색
        relevant_docs = self._search_relevant_documents(query, vector_db_data)

        if relevant_docs:
            return self._generate_rag_response(query, relevant_docs, vector_db_data)
        else:
            return self._get_no_match_response(query, len(vector_db_data))

    def _search_relevant_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """관련 문서 검색"""
        relevant_docs = []
        query_lower = query.lower()
        query_keywords = query_lower.split()

        for doc in documents:
            content = doc.get("content", "").lower()
            filename = doc.get("filename", "").lower()

            # 키워드 매칭 점수 계산
            content_matches = sum(1 for keyword in query_keywords if keyword in content)
            filename_matches = sum(1 for keyword in query_keywords if keyword in filename)

            total_score = content_matches * 2 + filename_matches  # 내용이 파일명보다 중요

            if total_score > 0:
                doc_with_score = doc.copy()
                doc_with_score['relevance_score'] = total_score
                relevant_docs.append(doc_with_score)

        # 관련도 순으로 정렬
        relevant_docs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return relevant_docs[:3]  # 상위 3개 문서

    def _generate_rag_response(self, query: str, relevant_docs: List[Dict], all_docs: List[Dict]) -> str:
        """RAG 기반 응답 생성"""
        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(relevant_docs[:2], 1):
            filename = doc.get("filename", f"문서{i}")
            content = doc.get("content", "")[:800]  # 800자 제한
            quality_score = doc.get("quality_score", 0)
            context_parts.append(f"📄 **{filename}** (품질점수: {quality_score}점)\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        # 응답 생성 (시뮬레이션 - 실제로는 LLM API 호출)
        response = f"""📚 **저장된 지식을 바탕으로 답변드리겠습니다.**

**질문:** {query}

**답변:**
{len(all_docs)}개의 저장된 문서를 검색한 결과, 다음과 같은 관련 정보를 찾았습니다:

{context}

**💡 요약:**
위 정보를 바탕으로 '{query}'에 대해 말씀드리면, 저장된 문서들에서 관련 내용을 확인할 수 있습니다. 더 구체적인 질문이 있으시면 언제든 물어보세요!

---
📊 **검색 결과:** {len(relevant_docs)}개 문서 매칭 | 💾 **전체 문서:** {len(all_docs)}개"""

        return response

    def _get_no_knowledge_response(self) -> str:
        """지식이 없을 때의 응답"""
        return """🤔 **아직 저장된 지식이 없습니다.**

현재 학습한 문서가 없어서 구체적인 답변을 드리기 어렵습니다.

**다음 단계를 권장드립니다:**
1. 📚 **지식등록** 메뉴에서 관련 문서를 업로드
2. 🤖 AI 분석을 통한 문서 보완
3. 💾 VectorDB에 저장

문서를 등록하신 후 다시 질문해주시면 더 정확한 답변을 드릴 수 있습니다! 😊"""

    def _get_no_match_response(self, query: str, total_docs: int) -> str:
        """매칭되는 문서가 없을 때의 응답"""
        return f"""🔍 **'{query}'에 대한 구체적인 정보를 찾지 못했습니다.**

현재 {total_docs}개의 문서가 저장되어 있지만, 질문과 직접적으로 관련된 내용을 찾을 수 없었습니다.

**다음을 시도해보세요:**
- 🔄 다른 키워드로 질문 변경
- 📚 관련된 새로운 문서 업로드
- 💬 더 구체적이거나 일반적인 질문으로 재시도

**예시 질문:**
- "문서에서 주요 내용은 무엇인가요?"
- "등록된 문서들의 요약을 알려주세요"
- "[특정 키워드]에 대해 설명해주세요"

도움이 필요하시면 언제든 말씀해주세요! 😊"""

    def get_chat_statistics(self) -> Dict:
        """채팅 통계 정보"""
        chat_history = st.session_state.get('chat_history', [])
        return {
            'total_messages': len(chat_history),
            'user_messages': len([msg for msg in chat_history if msg.get('role') == 'user']),
            'assistant_messages': len([msg for msg in chat_history if msg.get('role') == 'assistant']),
            'available_documents': len(st.session_state.get('vector_db', []))
        }