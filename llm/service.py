import os
from openai import AzureOpenAI
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class AzureAIFoundryClient:
    """Azure AI Foundry 모델을 위한 클라이언트"""

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """채팅 완성 API 호출"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Azure AI 모델 호출 실패: {str(e)}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 임베딩 생성"""
        try:
            response = self.client.embeddings.create(
                model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise Exception(f"임베딩 생성 실패: {str(e)}")

# 전역 클라이언트 인스턴스
llm_client = AzureAIFoundryClient()

def get_llm_client() -> AzureAIFoundryClient:
    """LLM 클라이언트 인스턴스 반환"""
    return llm_client

def analyze_document_content(content: str) -> Dict[str, Any]:
    """문서 내용 분석"""
    messages = [
        {
            "role": "system",
            "content": "당신은 문서 분석 전문가입니다. 주어진 문서를 분석하여 핵심 내용, 주제, 품질 점수를 제공해주세요."
        },
        {
            "role": "user",
            "content": f"다음 문서를 분석해주세요:\n\n{content}"
        }
    ]

    try:
        response = llm_client.chat_completion(messages, temperature=0.3)

        # 실제 분석 결과 파싱 로직 구현 필요
        return {
            "analysis": response,
            "quality_score": 85,  # 임시값, 실제로는 분석 결과에서 추출
            "main_topics": ["분석된 주제1", "분석된 주제2"],  # 임시값
            "summary": response[:200] + "..." if len(response) > 200 else response
        }
    except Exception as e:
        return {
            "analysis": f"분석 중 오류 발생: {str(e)}",
            "quality_score": 0,
            "main_topics": [],
            "summary": "분석 실패"
        }

def generate_rag_response(query: str, context_docs: List[str]) -> str:
    """RAG 기반 응답 생성"""
    context = "\n\n".join(context_docs)

    messages = [
        {
            "role": "system",
            "content": "당신은 지식 관리 시스템의 AI 어시스턴트입니다. 제공된 문서 맥락을 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요."
        },
        {
            "role": "user",
            "content": f"맥락:\n{context}\n\n질문: {query}"
        }
    ]

    try:
        return llm_client.chat_completion(messages, temperature=0.7)
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"