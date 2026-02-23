import sys
import os

# 💡 맥북(macOS) 딥러닝 라이브러리 멈춤(데드락) 방지 4대장
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["USE_TF"] = "0"      # TensorFlow 사용 원천 차단
os.environ["USE_TORCH"] = "1"   # PyTorch만 사용하도록 강제

import json
import numpy as np
# 프로젝트 최상위 폴더(AIcompliance)를 파이썬 경로에 강제로 추가하는 코드
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.db_connection import GraphDBConnection
# Cross-Encoder 리랭커 임포트 (pip install sentence-transformers 필요)
from sentence_transformers import CrossEncoder

class AIComplianceRAG:
    def __init__(self):
        db_conn = GraphDBConnection()
        self.graph = db_conn.get_graph()
        self.embeddings = db_conn.get_embeddings() 
        self.llm = db_conn.get_llm()
        
        print("⏳ Cross-Encoder 리랭킹 모델 로딩 중...")
        self.reranker = CrossEncoder('Dongjin-kr/ko-reranker') 
        
        # 💡 사전에 구축한 50개 QA JSON 셋 불러오기
        self.qa_dataset = self.load_qa_dataset()

    def load_qa_dataset(self) -> List[dict]:
        """JSON 파일로 저장된 50개의 QA 셋을 불러옵니다."""
        # 실제 JSON 파일이 위치한 경로로 맞춰주세요.
        file_path = os.path.join(os.path.dirname(__file__), 'qa_dataset.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ QA 데이터셋을 불러오지 못했습니다. 파일 위치를 확인하세요: {e}")
            return []

    def get_few_shot_examples(self, query: str, k: int = 2) -> str:
        """현재 질문과 가장 비슷한 QA 예시 2개를 찾아냅니다."""
        if not self.qa_dataset:
            return ""
        
        # 질문 간의 유사도를 비교하여 가장 비슷한 QA 셋을 찾음
        query_vector = self.embeddings.embed_query(query)
        scored_examples = []
        
        for qa in self.qa_dataset:
            qa_vector = self.embeddings.embed_query(qa['question'])
            # 코사인 유사도 계산
            similarity = np.dot(query_vector, qa_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(qa_vector))
            scored_examples.append((similarity, qa))
            
        # 가장 점수가 높은 상위 k개의 예시 추출
        top_examples = sorted(scored_examples, key=lambda x: x[0], reverse=True)[:k]
        
        few_shot_text = ""
        for i, (_, qa) in enumerate(top_examples):
            few_shot_text += f"\n[참고 예시 {i+1}]\nQ: {qa['question']}\nA: {qa['answer']}\n"
        
        return few_shot_text

    def analyze_and_route_query(self, query: str) -> List[str]:
        """💡 [Step 1] 질문의 복잡도를 판단하고 3가지 유형으로 라우팅/분할합니다."""
        router_prompt = ChatPromptTemplate.from_template("""
        당신은 기업의 AI 컴플라이언스 분석가입니다. 사용자의 질문을 분석하여 데이터베이스 검색에 최적화된 하위 질문(Sub-queries) 리스트로 변환하십시오.
        
        [분할 핵심 지침]
        - 사용자의 질문에 "벌칙", "벌금", "제재", "위반 시", "페널티", "리스크" 등의 단어가 포함되어 있다면, **반드시 그 제재 내용만을 묻는 명확한 하위 질문 1개를 리스트에 추가**해야 합니다. (예: "고위험 AI 시스템 위반 시 부과되는 벌칙과 과징금 규모")
        
        [분류 및 분할 기준]
        1. 단순 포괄형: "이거 문제 될까?" 형태의 일반적 질문 -> 핵심 키워드만 뽑아 1개의 질문으로 정리.
        2. 단일 심층형: 특정 조항이나 벌금을 깊게 묻는 질문 -> 깊게 파고들 수 있도록 1개의 구체적 질문으로 변환.
        3. 다중 분할형: 규제 대상 여부, 핵심 의무, 벌칙 등을 복합적으로 묻는 긴 질문 -> 2~3개의 개별 질문으로 분할. (예: ["채용 AI의 고위험 여부", "고위험 AI 위반 벌칙 금액"])
        
        반드시 파이썬 List 형태의 문자열(예: ["질문1", "질문2"])만 출력하십시오. 절대 다른 설명은 붙이지 마십시오.
        
        사용자 질문: {query}
        """)
        
        chain = router_prompt | self.llm | StrOutputParser()
        result_str = chain.invoke({"query": query}).strip()
        
        try:
            sub_queries = eval(result_str)
            if not isinstance(sub_queries, list):
                sub_queries = [query]
        except:
            sub_queries = [query]
            
        print(f"\n[🔀 질문 라우팅 및 분할 결과]: {sub_queries}")
        return sub_queries

    def translate_query(self, query: str) -> str:
        trans_prompt = ChatPromptTemplate.from_template(
            "Translate the following Korean query into English legal terms (EU AI Act style). "
            "Just output the translated text without any explanation.\nQuery: {query}"
        )
        chain = trans_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def retrieve_and_rerank_context(self, sub_queries: List[str], final_k: int = 5) -> str:
        """[Step 2] 벡터 검색 후 Cross-Encoder로 정밀하게 재정렬(Re-ranking)합니다."""
        
        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embedding', 12, $query_vector)
        YIELD node AS chunk, score
        MATCH (parent)-[:HAS_CHUNK]->(chunk)
        
        OPTIONAL MATCH (parent)-[r1]->(t1)
        WHERE type(r1) <> 'HAS_CHUNK' AND type(r1) <> 'INCLUDES'
        
        OPTIONAL MATCH (t1)-[r2]->(t2)
        WHERE type(r2) IN ['MANDATED_FOR', 'PERMITS', 'PENALIZES_WITH', 'APPLIES_TO', 'LEADS_TO', 'ENCOMPASSES', 'SUPPLEMENTS']
        
        WITH chunk, score, parent, r1, t1, r2, t2,
             coalesce(t1.name, t1.level, t1.id, labels(t1)[0]) AS t1_base,
             CASE 
                WHEN 'Sanction' IN labels(t1) THEN ' [금액: ' + coalesce(t1['amount'], '명시안됨') + ', 유형: ' + coalesce(t1['type'], '분류안됨') + ', 상세: ' + coalesce(t1['description'], '없음') + ']'
                WHEN 'Requirement' IN labels(t1) OR 'Support' IN labels(t1) THEN ' [상세: ' + coalesce(t1['description'], '없음') + ']'
                WHEN 'TechCriterion' IN labels(t1) OR 'UsageCriterion' IN labels(t1) THEN ' [기준값: ' + coalesce(t1['threshold_value'], '') + ' ' + coalesce(t1['unit'], '') + ']'
                WHEN 'Concept' IN labels(t1) THEN ' [언어: ' + coalesce(t1['lang'], '') + ']'
                ELSE ''
             END AS t1_extra,
             CASE
                WHEN 'Concept' IN labels(t2) THEN ' [언어: ' + coalesce(t2['lang'], '') + ']'
                WHEN 'Requirement' IN labels(t2) OR 'Support' IN labels(t2) OR 'Sanction' IN labels(t2) THEN ' [상세: ' + coalesce(t2['description'], '없음') + ']'
                ELSE ''
             END AS t2_extra
             
        WITH chunk, score, parent,
             CASE 
                WHEN t1 IS NOT NULL AND t2 IS NULL THEN type(r1) + " -> " + t1_base + t1_extra
                WHEN t1 IS NOT NULL AND t2 IS NOT NULL THEN type(r1) + " -> " + t1_base + t1_extra + " => [" + type(r2) + " -> " + coalesce(t2.name, t2.level, t2.id, labels(t2)[0]) + t2_extra + "]"
                ELSE null
             END AS path_info

        RETURN 
            chunk.text AS chunk_text,
            labels(parent)[0] AS parent_type,
            coalesce(parent.id, parent.name, 'ID없음') AS parent_id,
            coalesce(parent.title, '') AS parent_title,
            collect(DISTINCT path_info) AS target_info
        """
        
        unique_chunks = {}
        for sq in sub_queries:
            translated_query = self.translate_query(sq)
            query_vector = self.embeddings.embed_query(translated_query)
            results = self.graph.query(cypher_query, params={"query_vector": query_vector})
            
            for res in results:
                chunk_text = res['chunk_text']
                if chunk_text not in unique_chunks:
                    unique_chunks[chunk_text] = res

        if not unique_chunks:
            return "관련된 법적 근거를 찾을 수 없습니다."

        print(f"👉 1차 벡터 검색 완료: 총 {len(unique_chunks)}개의 조항 확보. 리랭킹 진행 중...")

        chunk_list = list(unique_chunks.values())
        pairs = [[sub_queries[0], item['chunk_text']] for item in chunk_list]
        scores = self.reranker.predict(pairs)
        
        scored_results = sorted(zip(scores, chunk_list), key=lambda x: x[0], reverse=True)
        top_results = [item for score, item in scored_results[:final_k]]

        print(f"🎯 리랭킹 완료: 가장 적합한 상위 {final_k}개 문서를 추렸습니다.")

        formatted_context = []
        for res in top_results:
            p_type = res['parent_type']
            p_id = res['parent_id']
            p_title = res['parent_title']
            valid_targets = [t for t in res['target_info'] if t]
            t_info = "\n  - " + "\n  - ".join(valid_targets) if valid_targets else "추가 연결된 규제 없음"
            title_str = f" ({p_title})" if p_title else ""
            
            context_piece = (
                f"--- 출처: [{p_type}] {p_id}{title_str} ---\n"
                f"내용(Chunk): {res['chunk_text']}\n"
                f"관련 구조(Graph): {t_info}\n"
            )
            formatted_context.append(context_piece)

        return "\n".join(formatted_context)

    def generate_answer(self, query: str, history: List[Dict] = None) -> Tuple[str, str]:
        
        # 이전 대화 내용을 하나의 문자열로 포맷팅
        history_text = ""
        if history:
            for msg in history:
                role = "사용자" if msg["role"] == "user" else "AI(이전 답변)"
                history_text += f"[{role}]: {msg['content']}\n"
        else:
            history_text = "이전 대화 없음."

        sub_queries = self.analyze_and_route_query(query)
        context = self.retrieve_and_rerank_context(sub_queries, final_k=8)
        few_shot_examples = self.get_few_shot_examples(query, k=2)

        prompt = ChatPromptTemplate.from_template("""
        당신은 포춘 500대 기업의 최고 AI 컴플라이언스 책임자(CCO)이자 글로벌 최고 수준의 법률 고문입니다.
        아래 제공된 [이전 대화 기록]의 문맥을 파악하고, 오직 [Context]에 있는 실제 법안 정보만을 바탕으로 [사용자 질문]에 대해 가장 전문적이고 깊이 있는 법률 컨설팅을 제공하십시오.

        [데이터 분석 및 출력 필수 지침 🚨]
        1. Context 엄수: [Context]의 '내용(Chunk)'과 '관련 구조(Graph)'에 있는 텍스트를 모두 분석하십시오. 특히 [상세:], [금액:], [유형:] 등의 구체적인 데이터는 절대 누락하지 마십시오. 금액이 없다면 "관련 법률상 구체적인 금액이 명시되어 있지 않으나, 법적 제재 리스크가 존재합니다"라고 명확히 기재하십시오.
        2. 전문적인 어투: 확신에 찬 전문가의 어투를 사용하며, 설명 중 반드시 "유럽 인공지능법(EU AI Act) 제O조에 따르면~"과 같이 구체적인 법적 근거를 텍스트에 자연스럽게 녹여내십시오.

        [답변 구조 및 포맷 지침 (절대 준수!)]
        당신의 답변은 반드시 아래 제시된 형태와 순서대로 출력되어야 합니다. (안내 문구나 괄호는 출력하지 마십시오.)

        (이 위치에 상세 본문 2문단 이상 작성)
        - 1문단: 해당 AI 시스템의 규제 대상 및 등급 분류 여부, 그리고 왜 그렇게 분류되는지(이유)를 Context를 바탕으로 상세히 설명하십시오.
        - 2문단: 기업이 시장 출시 전/후에 준수해야 할 '핵심 의무(투명성, 데이터 관리 등)'와 '위반 시 리스크'를 구체적으로 설명하십시오.

        (이 위치에 추가 정보 요청 작성 - 필수)
        - 사용자의 질문에 서비스 국가, 데이터 수집 범위(생체 데이터 등), 인간 개입 여부 중 하나라도 명확하지 않다면 반드시 아래와 같은 형태의 역질문을 생성하십시오.
        - 💡 중요: 질문이 포괄적일 경우, **반드시 2개 이상**의 구체적인 확인 사항을 불릿 포인트(`*`) 리스트로 작성하십시오.
        - 💡 중요: 절대 명사형 종결어미(예: '~여부')로 끝내지 마십시오. 반드시 자연스러운 대화형 질문 형태(예: "~하나요?", "~인가요?")로 끝내십시오.
        - 도입부 예시: "더욱 정확한 맞춤형 규제 리스크를 진단해 드리기 위해, 추가로 확인이 필요한 사항들이 있습니다. 혹시 아래 내용에 대해 조금 더 자세히 알려주실 수 있을까요?"

        ---
        [요약 및 참고]
        - 규제 대상: (핵심만 명사형으로 요약)
        - 핵심 의무: (핵심만 명사형으로 요약)
        - 위반 리스크: (핵심만 명사형으로 요약)
        - 근거 조항: (참조한 법안 및 조항 번호)
        
        [이전 대화 기록]
        {history_text}

        [참고 예시 (어투와 논리 전개 방식을 참고하십시오)]
        {few_shot_examples}
        
        [실제 법안 데이터 (Context)]
        {context}

        [사용자 질문 (Question)]
        {query}
        """)

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "history_text": history_text,  # 💡 프롬프트 변수에 주입
            "context": context, 
            "query": query,
            "few_shot_examples": few_shot_examples if few_shot_examples else "별도 예시 없음."
        })

        if answer.startswith("답변"):
            answer = answer.split("\n", 1)[-1].strip()

        return answer, context

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("⚙️ AI Compliance Advanced RAG 엔진 가동 중...")
    rag = AIComplianceRAG()
    
    test_question = "한국 AI 기본법에서 말하는 '고영향 AI' 기준이랑, 유럽 AI Act의 고위험 AI 기준이 어떻게 연결돼? 한국에서 서비스할 때 유럽 법안의 어떤 부분을 보완해서 참고해야 할까?"
    #"유럽에서 신입사원 채용 면접을 분석하는 AI 시스템을 도입하려고 해. 이거 고위험 AI에 해당해? 그리고 위반하면 어떤 벌칙이 있어?"
    
    answer, context = rag.generate_answer(test_question)
    
    print("\n[최종 답변]")
    print(answer)