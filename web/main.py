"""
main.py - FastAPI 백엔드 서버
AI Compliance RAG 시스템의 API 엔드포인트를 제공합니다.
"""

import sys
import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 프로젝트 루트 경로를 sys.path에 추가 (상대 임포트 지원)
# 레포 최상위 경로를 추가 (src 패키지 접근용)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# rag_engine은 무거운 모델(Cross-Encoder)을 로드하므로
# 앱 시작 시 한 번만 초기화하여 전역 변수로 관리합니다.
rag_instance = None


# ─────────────────────────────────────────
# Lifespan: 앱 시작/종료 시 RAG 엔진 초기화
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 RAG 엔진을 한 번만 로드합니다."""
    global rag_instance
    print("🚀 서버 시작: AI Compliance RAG 엔진을 초기화합니다...")
    try:
        from src.rag_engine import AIComplianceRAG
        rag_instance = AIComplianceRAG()
        print("✅ RAG 엔진 초기화 완료!")
    except Exception as e:
        print(f"❌ RAG 엔진 초기화 실패: {e}")
        # 초기화 실패 시에도 서버는 기동되지만, /chat 호출 시 에러 반환
    yield
    # 종료 시 정리 로직이 필요하면 여기에 추가
    print("🛑 서버 종료.")


# ─────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────
app = FastAPI(
    title="AI Compliance Assistant API",
    description="EU AI Act 기반 AI 컴플라이언스 분석 및 답변 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정 (Streamlit 프론트엔드와의 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 운영 환경에서는 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Request / Response 스키마 정의
# ─────────────────────────────────────────
class HistoryMessage(BaseModel):
    """대화 히스토리 단일 메시지 스키마"""
    role: str      # "user" 또는 "assistant"
    content: str


class ChatRequest(BaseModel):
    """사용자 질문 요청 스키마"""
    question: str
    history: Optional[List[HistoryMessage]] = []   # 이전 대화 기록 (없으면 빈 리스트)


class SourceReference(BaseModel):
    """참조 법적 근거 스키마"""
    source_type: str          # 예: Article, Recital, Annex 등
    source_id: str            # 예: Article_6, Recital_47
    title: Optional[str] = "" # 조항 제목 (있을 경우)
    excerpt: str              # 관련 조문 발췌


class ChatResponse(BaseModel):
    """AI 답변 응답 스키마"""
    answer: str
    sources: List[SourceReference]
    sub_queries: Optional[List[str]] = []


# ─────────────────────────────────────────
# 헬퍼: raw context 문자열 → SourceReference 리스트 파싱
# ─────────────────────────────────────────
def parse_sources(raw_context: str) -> List[SourceReference]:
    """
    retrieve_and_rerank_context()가 반환하는 포맷:
    --- 출처: [ParentType] parent_id (title) ---
    내용(Chunk): ...
    관련 구조(Graph): ...
    를 파싱하여 SourceReference 리스트로 변환합니다.
    """
    sources: List[SourceReference] = []
    # 각 블록은 "--- 출처:" 로 시작
    blocks = raw_context.split("--- 출처:")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        # 첫 줄: "[ParentType] parent_id (title) ---"
        header_line = lines[0].replace("---", "").strip()
        # 내용(Chunk) 추출
        chunk_text = ""
        for line in lines[1:]:
            if line.startswith("내용(Chunk):"):
                chunk_text = line.replace("내용(Chunk):", "").strip()
                break

        # [Type] id (title) 파싱
        type_match = re.match(r"\[(.+?)\]\s+(\S+)\s*(?:\((.+?)\))?", header_line)
        if type_match:
            src_type = type_match.group(1)
            src_id = type_match.group(2)
            src_title = type_match.group(3) or ""
        else:
            src_type = "Unknown"
            src_id = header_line[:50]
            src_title = ""

        # 동일 출처 중복 제거
        if not any(s.source_id == src_id for s in sources):
            sources.append(SourceReference(
                source_type=src_type,
                source_id=src_id,
                title=src_title,
                excerpt=chunk_text[:300] + ("..." if len(chunk_text) > 300 else ""),
            ))
    return sources


# ─────────────────────────────────────────
# API 엔드포인트
# ─────────────────────────────────────────
@app.get("/health")
async def health_check():
    """서버 상태 및 RAG 엔진 초기화 여부 확인"""
    return {
        "status": "ok",
        "rag_engine_ready": rag_instance is not None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    [메인 엔드포인트]
    1. analyze_and_route_query() → 질문 유형 분류 및 서브쿼리 생성
    2. retrieve_and_rerank_context() → 벡터 검색 + Cross-Encoder 리랭킹
    3. generate_answer() → Few-shot 프롬프팅 기반 답변 생성
    4. 답변 + 참조 법적 근거 JSON 반환
    """
    if rag_instance is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 엔진이 초기화되지 않았습니다. 서버 로그를 확인해주세요.",
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    try:
        # ── history를 dict 리스트로 변환하여 rag_engine에 직접 전달
        # 검색(retrieve)은 현재 question만으로, 답변 생성(LLM)에서만 history 활용
        history_dicts = [{"role": h.role, "content": h.content} for h in request.history]

        # ── Step 2 & 3: 검색 → 리랭킹 → 답변 생성
        answer, raw_context = rag_instance.generate_answer(question, history=history_dicts)

        # ── Step 4: context 파싱 → 참조 출처 추출
        sources = parse_sources(raw_context)

        # 서브쿼리 정보도 반환 (프론트에서 "분석 과정" 노출 가능)
        sub_queries = rag_instance.analyze_and_route_query(question)

        return ChatResponse(
            answer=answer,
            sources=sources,
            sub_queries=sub_queries,
        )

    except Exception as e:
        print(f"❌ /chat 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}")


# ─────────────────────────────────────────
# 직접 실행 시 uvicorn 서버 기동
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # 개발 모드: 코드 변경 시 자동 재시작
        log_level="info",
    )
