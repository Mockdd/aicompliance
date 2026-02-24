import sys
import os

# 💡 맥북(macOS) 딥러닝 라이브러리 멈춤(데드락) 방지 4대장
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["USE_TF"] = "0"      # TensorFlow 사용 원천 차단
os.environ["USE_TORCH"] = "1"   # PyTorch만 사용하도록 강제

import json
import numpy as np
# 프로젝트 최상위 폴더를 파이썬 경로에 강제로 추가하는 코드
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
        
        # 💡 사전에 구축한 QA JSON 셋 불러오기
        self.qa_dataset = self.load_qa_dataset()

    def load_qa_dataset(self) -> List[dict]:
        """JSON 파일로 저장된 QA 셋을 불러옵니다."""
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
        
        query_vector = self.embeddings.embed_query(query)
        scored_examples = []
        
        for qa in self.qa_dataset:
            # 💡 [커닝 방지]: 지금 풀고 있는 시험 문제와 똑같은 질문은 참고 예시에서 제외!
            if qa['question'].strip() == query.strip():
                continue
                
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
        # 💡 [핵심 수술 1]: 단순 번역을 넘어, 영어 공식 법률 용어(Emotion Recognition 등)를 덧붙여 검색력을 폭발시킵니다!
        trans_prompt = ChatPromptTemplate.from_template("""
        Translate the following Korean query into English legal terms suitable for the EU AI Act and Korea AI Act.
        Also, strongly expand the query by adding relevant official legal keywords (e.g., 'Emotion Recognition', 'Prohibited practice', 'Biometric', 'High-Risk', 'Penalty', 'Fines').
        Just output the translated text and expanded keywords without any explanation.
        
        Query: {query}
        """)
        chain = trans_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def retrieve_and_rerank_context(self, query: str, sub_queries: List[str], final_k: int = 10) -> str:
        """[Step 2] 벡터 검색 후 Cross-Encoder로 정밀하게 재정렬(Re-ranking)합니다."""
        
        # 💡 [핵심 수술 2]: 1차 그물망을 12개 -> 30개로 대폭 늘려서 구석에 박힌 페널티 조항까지 다 끌어옵니다!
        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embedding', 30, $query_vector)
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
        translated_sq_list = [] # 리랭커에게 넘겨줄 영어 키워드 수집통
        
        for sq in sub_queries:
            translated_query = self.translate_query(sq)
            translated_sq_list.append(translated_query) # 수집
            
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
        
        # 💡 [핵심 수술 3]: 리랭커에게 '원본 한국어' + '번역/확장된 영어 법률 용어'를 모두 넘겨주어 채점 정확도를 극대화합니다.
        combined_rerank_query = query + "\n" + "\n".join(translated_sq_list)
        pairs = [[combined_rerank_query, item['chunk_text']] for item in chunk_list]
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
        
        history_text = ""
        if history:
            for msg in history:
                role = "사용자" if msg["role"] == "user" else "AI(이전 답변)"
                history_text += f"[{role}]: {msg['content']}\n"
        else:
            history_text = "이전 대화 없음."

        sub_queries = self.analyze_and_route_query(query)
        
        # 💡 넉넉하게 10개의 핵심 조항을 가져옵니다.
        context = self.retrieve_and_rerank_context(query, sub_queries, final_k=10)
        few_shot_examples = self.get_few_shot_examples(query, k=2)

        prompt = ChatPromptTemplate.from_template("""
        당신은 기업의 최고 AI 컴플라이언스 책임자(CCO)입니다.
        오직 아래 제공된 [Context]의 내용만을 사용하여 [사용자 질문]에 대해 최고 수준의 법률 컨설팅 답변을 작성하십시오. 당신이 학습한 사전 지식은 절대 사용하지 마십시오.

        [데이터 분석 및 출력 필수 지침 🚨]
        1. 내용 창조 절대 금지: 제공된 데이터에 명시적으로 존재하지 않는 법안명, 조항 번호, 금액, 특정 의무 사항, 벌칙은 절대 지어내지 마십시오.
        2. 모순된 정보 출력 금지: 질문에서 묻는 내용이 제공된 데이터에 없다면 억지로 지어내지 마십시오. 반대로, 정보가 존재하여 작성해 놓고 그 옆에 "(명시되지 않음)"이나 "(확인 불가)"라는 말을 동시에 적는 바보 같은 짓을 절대 하지 마십시오.
        3. 예시 베끼기 금지: [참고 예시]는 오직 '어투와 전개 방식'만 참고하고, 예시의 내용은 절대 베끼지 마십시오.
        4. 기계 말투 절대 금지 🚨: 답변을 작성할 때 화면에 절대 "[Context]", "제공된 데이터", "검색된 문서" 같은 시스템적인 단어를 출력하지 마십시오. 정보가 없을 경우 반드시 "관련 법률상 명시되어 있지 않습니다" 또는 "현행 규정에서는 확인되지 않습니다"라고 실제 사람(법률 고문)처럼 자연스럽게 답변하십시오.

        [답변 구조 및 포맷 지침]
        아래 형식과 순서를 엄격히 지켜 답변하십시오. (주의: '본문', '1문단' 같은 안내용 괄호나 단어는 화면에 절대 출력하지 마십시오.)

        (이 위치에 자연스러운 줄글 형태로 상세 답변을 작성하십시오. 🚨 절대 단락 시작 부분에 '-', '*', 숫자 등의 기호를 붙이지 말고, 일반적인 보고서/에세이 산문 형태로 작성하십시오.)
        질문에 대한 '핵심 결론'을 가장 먼저 명쾌하게 제시하는 두괄식으로 시작하십시오.
        질문이 두 개 이상의 법안(예: 한국 AI 기본법과 유럽 AI Act)의 관계나 연결성을 묻는다면, 먼저 각 법안의 기준을 구체적으로 설명하십시오. 그 후, 이들이 어떻게 연결되는지, 실무적으로 서비스 도입 시 어떤 부분을 상호 보완해야 하는지 논리적으로 분석해 답변하십시오.
        해당 AI 시스템의 '규제 대상 및 등급 분류 여부', 기업이 지켜야 할 '핵심 의무', '위반 리스크'를 필수적으로 포함하여 설명하십시오.

        (줄바꿈 후 역질문 작성 - 필수)
        더욱 정확한 맞춤형 규제 리스크를 진단하기 위해, 추가로 확인이 필요한 사항은 다음과 같습니다.
        * (구체적인 확인 사항 1 - 반드시 '~인가요?', '~합니까?' 등 자연스러운 대화형/질문형 종결어미 사용)
        * (구체적인 확인 사항 2 - 반드시 '~인가요?', '~합니까?' 등 자연스러운 대화형/질문형 종결어미 사용)

        (줄바꿈 후 요약 작성 - 필수)
        - 규제 대상: 확인된 핵심 요약
        - 핵심 의무: 확인된 핵심 요약 (정보가 없을 경우에만 "관련 법률상 명시되어 있지 않음"이라고 기재)
        - 위반 리스크: 확인된 핵심 요약 (정보가 없을 경우에만 "관련 법률상 명시되어 있지 않음"이라고 기재)
        - 근거 조항: 확인된 등장 법안 및 조항 번호 (정보가 없을 경우에만 "관련 법률상 명시되어 있지 않음"이라고 기재)
        💡 주의: 요약 내용에 법안명이나 조항을 이미 작성해 놓고 그 옆에 "(명시되지 않음)"을 덧붙이는 행위를 절대 금지합니다.
        
        [이전 대화 기록]
        {history_text}

        [참고 예시 (어투와 전문적인 형식만 참고할 것. 내용 베끼기 절대 금지!)]
        {few_shot_examples}
        
        [실제 법안 데이터 (Context)]
        {context}

        [사용자 질문 (Question)]
        {query}
        """)

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "history_text": history_text,
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
    
    test_question = "유럽 지사 채용 과정에서 지원자의 표정과 목소리 톤을 분석해서 스트레스 저항성을 평가하는 AI 면접 툴을 전면 도입하려고 합니다. 위반 시 페널티와 미리 준비해야 할 사항이 궁금합니다."
    
    answer, context = rag.generate_answer(test_question)
    
    print("\n[최종 답변]")
    print(answer)