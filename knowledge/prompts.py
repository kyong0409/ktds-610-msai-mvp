ANALYZE_DOCUMENT_PROMPT = """
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

GENERATE_ENHANCED_KNOWLEDGE_DOCUMENT_PROMPT = """
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

KNOTE_TO_STANDARD_DOC_PROMPT = """
[역할]
당신은 IT 회사의 지식 관리 전문가이자 기술 문서 편집자입니다.
입력된 K-Note(JSON)를 분석하여, 사내 게시판에 게시 가능한 "표준 지식 문서" 형식으로 재구성합니다.

[입력]
1) K-Note(JSON)
{k_note_json}

2) (선택) 추가 보완사항(Improvement Points)
{additional_improvement_points}

[변환 규칙: K-Note → 표준 문서 매핑]
- 문서 제목         ← k_note.title (없으면 k_note.proposal 요약으로 생성)
- 작성자            ← k_note.owners (없으면 "미확인")
- 작성일            ← 미확인 (입력에 명시된 경우만 사용; 임의 생성 금지)
- 버전              ← k_note.version (없으면 "미확인")
- 프로젝트/적용 분야 ← k_note.applicability.when / when_not / assumptions를 요약하여 기술 영역과 적용 조건으로 정리
- 주요 태그          ← title, proposal, applicability에서 핵심 키워드 5~10개 추출 (약어/동의어 정규화, 중복 제거)

[본문 구성 규칙]
1. 목적(Purpose)
   - k_note.proposal을 한 줄 핵심 문장 + 2~3문장 보충으로 기술
2. 배경 및 문제 정의(Background / Problem Statement)
   - applicability.when / when_not / assumptions를 바탕으로 문제 상황, 전제, 적용/비적용 조건을 구조적으로 정리
3. 접근 방법 및 절차(Approach / Methodology)
   - 제안(statement) 핵심 아이디어와 단계별 절차(필요 인프라, 메시지 흐름, 데이터 모델/일관성, 운영 가이드)를 항목화
4. 결과 및 성과(Results / Outcomes)
   - k_note.metrics_effect 내용을 정량·정성으로 요약(지표명/변화량/단위 명시)
5. 한계 및 교훈(Limitations / Lessons Learned)
   - k_note.risks_limits를 근거와 함께 정리
6. 적용 및 재사용 방안(Application / Reusability)
   - k_note.recommended_experiments를 "권장 실험/적용 절차"로 재구성(환경, KPI, 성공 기준 포함)
   - 재사용 체크리스트(필수 전제/주의사항) 포함
7. 참고 자료(References)
   - k_note.evidence 배열을 인용 리스트로 변환:
     - 형식: [doc_id#chunk_id] "quote" (confidence: 0.00~1.00)
   - 출처가 부족하면 "추가 출처 필요"로 표시(창작 금지)

[보완사항 생성 규칙]
- {additional_improvement_points}가 비어 있으면, 다음을 근거로 보완점을 자동 도출:
  1) risks_limits → 위험 완화/운영 가드레일/모니터링 항목
  2) evidence.confidence < 0.75 → 근거 보강 필요 영역
  3) recommended_experiments → 검증 항목/추가 실험 계획
- 보완점은 실행 가능하도록 "카테고리, 이슈, 제안, 근거(있으면)" 형태의 불릿으로 작성

[스타일 가이드]
- 명확·간결·재사용 친화 용어 사용, 불필요한 수식·과장 금지
- 무근거한 신규 사실 창작 금지(입력에 없는 수치·날짜·저자 생성 금지)
- 목록은 5~8개 이내로, 과도하면 "(… 외 N건)"으로 축약
- 약어는 첫 등장 시 풀어쓴 뒤 괄호로 약어 병기(예: Command Query Responsibility Segregation(CQRS))

[출력 형식: 표준 지식 문서]
### 메타데이터
- 문서 제목:
- 작성자:
- 작성일:
- 버전:
- 프로젝트/적용 분야:
- 주요 태그(키워드):

### 문서 본문
1. 목적 (Purpose)
2. 배경 및 문제 정의 (Background / Problem Statement)
3. 접근 방법 및 절차 (Approach / Methodology)
4. 결과 및 성과 (Results / Outcomes)
5. 한계 및 교훈 (Limitations / Lessons Learned)
6. 적용 및 재사용 방안 (Application / Reusability)
7. 참고 자료 (References)

### 보완이 필요한 점 (Improvement Points)
- 카테고리: … / 이슈: … / 제안: … / 근거: …

[검증]
- 출력에 임의 생성된 출처·수치·날짜가 포함되면 안 됩니다.
- 모든 주장에는 K-Note 입력 내 항목을 근거로 연결하거나 "추정(근거: …)"로 명시하세요.
"""