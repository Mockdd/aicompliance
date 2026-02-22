"""
app.py - AI Compliance Assistant Streamlit UI
라이트모드 / 블루 컨셉 / 카카오톡형 채팅 인터페이스
"""

import streamlit as st
import requests
from datetime import datetime
from typing import List, Dict

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Compliance Checker",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CSS - 라이트모드 / 블루 컨셉
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

html, body, .stApp {
    background-color: #F4F7FD !important;
    font-family: 'Pretendard', -apple-system, sans-serif;
    color: #1a1f36;
}

/* ── 사이드바 ── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F8;
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ── 메인 컨텐츠 ── */
.block-container {
    padding: 2rem 2rem 6rem 2rem !important;
    max-width: 900px !important;
}

/* ── 헤더 ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 0 0 1.5rem 0;
    border-bottom: 2px solid #E2E8F8;
    margin-bottom: 2rem;
}
.app-header-icon {
    width: 54px; height: 54px;
    background: linear-gradient(135deg, #1B4FD8, #3B82F6);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    box-shadow: 0 4px 16px rgba(27,79,216,0.28);
    flex-shrink: 0;
}
.app-header h1 {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #1B4FD8 !important;
    letter-spacing: -0.5px;
    margin: 0 !important;
    line-height: 1.2 !important;
}
.app-header p {
    font-size: 0.84rem;
    color: #64748B;
    margin: 3px 0 0 0;
}

/* ── 채팅 래퍼 ── */
.chat-wrapper { display: flex; flex-direction: column; gap: 18px; }

/* ── 사용자 버블 ── */
.msg-user-row {
    display: flex; justify-content: flex-end;
    align-items: flex-end; gap: 8px;
}
.msg-user-bubble {
    background: linear-gradient(135deg, #1B4FD8, #2563EB);
    color: #FFFFFF;
    border-radius: 20px 20px 4px 20px;
    padding: 13px 18px;
    max-width: 68%;
    font-size: 0.93rem; line-height: 1.65;
    box-shadow: 0 3px 12px rgba(27,79,216,0.22);
    word-break: break-word;
}
.msg-time {
    font-size: 0.7rem; color: #94A3B8;
    white-space: nowrap; margin-bottom: 3px;
}

/* ── AI 버블 ── */
.msg-ai-row {
    display: flex; justify-content: flex-start;
    align-items: flex-start; gap: 10px;
}
.msg-ai-avatar {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #1B4FD8, #3B82F6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0; margin-top: 2px;
    box-shadow: 0 2px 8px rgba(27,79,216,0.2);
}
.msg-ai-content { max-width: 76%; }
.msg-ai-name {
    font-size: 0.74rem; font-weight: 700;
    color: #1B4FD8; margin-bottom: 5px; letter-spacing: 0.3px;
}
.msg-ai-bubble {
    background: #FFFFFF; color: #1E293B;
    border-radius: 4px 20px 20px 20px;
    padding: 15px 18px;
    font-size: 0.93rem; line-height: 1.75;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 1px solid #E2E8F8;
    word-break: break-word;
}

/* ── 근거 카드 ── */
.source-label {
    font-size: 0.77rem; font-weight: 700;
    color: #1B4FD8; letter-spacing: 0.4px;
    text-transform: uppercase; margin: 10px 0 7px 0;
}
.source-card {
    background: #F8FAFF;
    border: 1px solid #DBEAFE;
    border-left: 4px solid #1B4FD8;
    border-radius: 8px;
    padding: 10px 14px; margin-bottom: 7px;
}
.source-card-header {
    display: flex; align-items: center;
    gap: 8px; margin-bottom: 4px;
}
.source-tag {
    background: #DBEAFE; color: #1B4FD8;
    font-size: 0.71rem; font-weight: 700;
    padding: 2px 7px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.source-id { font-weight: 700; color: #1E293B; font-size: 0.84rem; }
.source-title-text { font-size: 0.77rem; color: #64748B; }
.source-excerpt {
    font-size: 0.81rem; color: #475569; line-height: 1.55;
    border-top: 1px solid #E2E8F8;
    padding-top: 6px; margin-top: 5px; font-style: italic;
}

/* ── 서브쿼리 칩 ── */
.sq-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 4px 0; }
.sq-chip {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    color: #1D4ED8; font-size: 0.77rem;
    padding: 3px 10px; border-radius: 20px; font-weight: 500;
}

/* ── 버튼 스타일 ── */
.stButton > button {
    background: #FFFFFF !important;
    color: #1B4FD8 !important;
    border: 1.5px solid #BFDBFE !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    line-height: 1.4 !important;
}
.stButton > button:hover {
    background: #EFF6FF !important;
    border-color: #1B4FD8 !important;
    box-shadow: 0 2px 8px rgba(27,79,216,0.1) !important;
}

/* ── 채팅 입력창 ── */
.stChatInput > div {
    border: 2px solid #BFDBFE !important;
    border-radius: 14px !important;
    background: #FFFFFF !important;
    box-shadow: 0 2px 12px rgba(27,79,216,0.08) !important;
}
.stChatInput > div:focus-within {
    border-color: #1B4FD8 !important;
    box-shadow: 0 2px 18px rgba(27,79,216,0.15) !important;
}

/* ── expander ── */
details { background: transparent !important; border: none !important; }
details summary {
    font-size: 0.82rem !important; font-weight: 600 !important;
    color: #1B4FD8 !important; padding: 6px 0 !important;
}

/* ── 빈 상태 ── */
.empty-state {
    text-align: center; padding: 64px 20px; color: #94A3B8;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 14px; }
.empty-state h3 {
    font-size: 1.1rem; font-weight: 700;
    color: #64748B; margin-bottom: 8px;
}
.empty-state p { font-size: 0.87rem; line-height: 1.65; }

/* ── 사이드바 로고 ── */
.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E2E8F8;
    margin-bottom: 1.2rem;
}
.sb-logo-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #1B4FD8, #3B82F6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.sb-logo-name { font-size: 0.95rem; font-weight: 800; color: #1B4FD8; }
.sb-logo-sub  { font-size: 0.71rem; color: #94A3B8; }

hr { border-color: #E2E8F8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────
def build_history_payload() -> List[Dict]:
    """
    session_state.messages에서 role/content만 추려
    API payload용 history 리스트를 만듭니다.
    현재 진행 중인 질문(마지막 user 메시지)은 제외하고 이전 대화만 포함합니다.
    """
    history = []
    messages = st.session_state.messages
    # 마지막 메시지는 방금 추가한 user 질문이므로 제외
    for msg in messages[:-1]:
        role = msg["role"]
        content = msg["content"]
        if role in ("user", "assistant"):
            history.append({"role": role, "content": content})
    return history


def call_chat(question: str) -> Dict:
    history = build_history_payload()
    try:
        r = requests.post(
            CHAT_ENDPOINT,
            json={"question": question, "history": history},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "⏱️ 응답 시간 초과. 잠시 후 다시 시도해주세요."}
    except requests.exceptions.ConnectionError:
        return {"error": "🔌 서버에 연결할 수 없습니다. FastAPI 서버 실행 여부를 확인해주세요."}
    except Exception as e:
        return {"error": f"❌ 오류: {str(e)}"}


def now_str() -> str:
    return datetime.now().strftime("%H:%M")


def render_sources_html(sources: List[Dict]) -> str:
    if not sources:
        return ""
    html = '<div class="source-label">📋 참조 법적 근거</div>'
    for src in sources:
        title = f'<span class="source-title-text"> — {src.get("title","")}</span>' if src.get("title") else ""
        excerpt = src.get("excerpt", "")
        exc_html = f'<div class="source-excerpt">"{excerpt}"</div>' if excerpt else ""
        html += f"""
        <div class="source-card">
            <div class="source-card-header">
                <span class="source-tag">{src.get("source_type","")}</span>
                <span class="source-id">{src.get("source_id","")}</span>{title}
            </div>{exc_html}
        </div>"""
    return html


def render_subqueries_html(sqs: List[str]) -> str:
    if not sqs:
        return ""
    chips = "".join(f'<span class="sq-chip">🔍 {q}</span>' for q in sqs)
    return f'<div class="sq-wrap">{chips}</div>'


def render_message(msg: Dict):
    role = msg["role"]
    content = msg["content"]
    t = msg.get("time", "")

    if role == "user":
        st.markdown(f"""
        <div class="msg-user-row">
            <span class="msg-time">{t}</span>
            <div class="msg-user-bubble">{content}</div>
        </div>""", unsafe_allow_html=True)

    else:
        sources = msg.get("sources", [])
        sub_queries = msg.get("sub_queries", [])

        st.markdown(f"""
        <div class="msg-ai-row">
            <div class="msg-ai-avatar">🪄</div>
            <div class="msg-ai-content">
                <div class="msg-ai-name">AI Compliance Assistant</div>
                <div class="msg-ai-bubble">{content}</div>
                <div class="msg-time" style="margin-top:5px;">{t}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        if sources or sub_queries:
            label = f"📚 법적 근거 {len(sources)}건" + (" · 분석 과정" if sub_queries else "")
            with st.expander(label):
                if sub_queries:
                    st.markdown("**🔀 질문 분석 (서브쿼리)**")
                    st.markdown(render_subqueries_html(sub_queries), unsafe_allow_html=True)
                    if sources:
                        st.markdown("---")
                if sources:
                    st.markdown(render_sources_html(sources), unsafe_allow_html=True)


# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🪄</div>
        <div>
            <div class="sb-logo-name">AI Compliance</div>
            <div class="sb-logo-sub">AI 규제 검토 챗봇</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<span style='font-size:0.82rem;font-weight:700;color:#374151;'>💡 예시 질문</span>",
                unsafe_allow_html=True)
    st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)

    examples = [
        "채용 AI가 고위험으로 분류되나요?",
        "생체인식 AI의 EU 내 사용 조건은?",
        "고위험 AI 시스템의 적합성 평가 절차",
        "AI 규정 위반 시 최대 과징금은?",
        "범용 AI(GPAI) 모델의 의무 사항",
        "AI 리터러시 의무는 누구에게 적용되나요?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:14]}", use_container_width=True):
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.button("🗑️ 대화 초기화", use_container_width=True, key="clear"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style='font-size:0.74rem;color:#94A3B8;line-height:1.65;margin-top:8px;'>
    본 서비스는 EU AI Act 및 한국 AI 기본법 기반 정보 제공 목적으로 운영됩니다.<br>
    법적 효력이 있는 공식 법률 자문이 아닙니다.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 메인 헤더
# ─────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">🪄</div>
    <div>
        <h1>AI COMPLIANCE CHECKER</h1>
        <p>EU AI Act · 한국 AI 기본법 기반 규제 검토 · 의무 사항 안내 · 법적 리스크 평가</p>
    </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 대화 이력 출력
# ─────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">⚖️</div>
        <h3>무엇이든 질문하세요</h3>
        <p>EU AI Act 및 한국 AI 기본법 관련 규제 여부, 의무 사항, 위반 시 벌칙 등<br>
        AI 컴플라이언스에 관한 질문에 법적 근거와 함께 답변해드립니다.</p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# 채팅 입력
# ─────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
placeholder_text = prefill if prefill else "규제가 궁금한 AI 시스템에 대해 질문하세요..."
user_input = st.chat_input(placeholder_text)

# 예시 버튼 클릭 처리
if prefill and not user_input:
    user_input = prefill


# ─────────────────────────────────────────
# 메시지 처리 & API 호출
# ─────────────────────────────────────────
if user_input and user_input.strip():
    question = user_input.strip()

    # 사용자 메시지 저장
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "time": now_str(),
    })

    # API 호출
    with st.spinner("법적 근거를 분석하는 중..."):
        result = call_chat(question)

    if "error" in result:
        answer = result["error"]
        sources, sub_queries = [], []
    else:
        answer = result.get("answer", "답변을 생성하지 못했습니다.")
        sources = result.get("sources", [])
        sub_queries = result.get("sub_queries", [])

    # AI 메시지 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "sub_queries": sub_queries,
        "time": now_str(),
    })

    st.rerun()