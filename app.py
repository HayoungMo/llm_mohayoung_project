from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import os

try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", str(st.secrets["GEMINI_API_KEY"]))
    if "GEMINI_MODEL" in st.secrets:
        os.environ.setdefault("GEMINI_MODEL", str(st.secrets["GEMINI_MODEL"]))
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", str(st.secrets["GROQ_API_KEY"]))
    if "GROQ_MODEL" in st.secrets:
        os.environ.setdefault("GROQ_MODEL", str(st.secrets["GROQ_MODEL"]))
except Exception:
    pass

from image_predictor import FurnitureImagePredictor
from rag_core import FurnitureRAG


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "data" / "furniture_knowledge.md"


st.set_page_config(
    page_title="가구 추천 AI",
    page_icon="house",
    layout="wide",
)


APP_CSS = """
<style>
    :root {
        --blue-900: #071a46;
        --blue-800: #0b2e75;
        --blue-700: #1054c9;
        --blue-600: #1677ff;
        --blue-500: #2396ff;
        --cyan-400: #22d3ee;
        --sky-100: #eaf6ff;
        --ink: #07111f;
        --muted: #667085;
        --line: rgba(44, 104, 196, 0.18);
        --glass: rgba(255, 255, 255, 0.78);
        --glass-strong: rgba(255, 255, 255, 0.92);
        --shadow: 0 22px 70px rgba(8, 36, 92, 0.15);
        --glow: 0 0 0 1px rgba(35, 150, 255, 0.20), 0 18px 45px rgba(22, 119, 255, 0.20);
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(circle at 8% 8%, rgba(34, 211, 238, 0.20), transparent 30%),
            radial-gradient(circle at 92% 18%, rgba(22, 119, 255, 0.24), transparent 34%),
            radial-gradient(circle at 52% 102%, rgba(16, 84, 201, 0.12), transparent 36%),
            linear-gradient(135deg, #f7fbff 0%, #eef7ff 42%, #f8fbff 100%);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 5.2rem !important;
        padding-bottom: 4rem;
    }

    h1, h2, h3, p, span, div {
        letter-spacing: 0;
    }

    .app-header {
        display: grid;
        grid-template-columns: 1fr;
        align-items: center;
        gap: 0;
        min-height: 76px;
        padding: 22px 26px;
        margin: 10px 0 32px;
        border: 1px solid rgba(35, 150, 255, 0.18);
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.72);
        box-shadow: 0 18px 45px rgba(8, 36, 92, 0.10);
        backdrop-filter: blur(18px);
        position: relative;
        overflow: hidden;
    }

    .app-header::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, rgba(22, 119, 255, 0.16), transparent 35%, rgba(34, 211, 238, 0.16));
        pointer-events: none;
    }

    .brand {
        position: relative;
        font-size: 29px;
        line-height: 1.15;
        font-weight: 950;
        color: var(--blue-900);
        white-space: nowrap;
    }

    .brand::after {
        content: "";
        display: block;
        width: 54px;
        height: 4px;
        margin-top: 7px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--blue-600), var(--cyan-400));
        box-shadow: 0 0 18px rgba(35, 150, 255, 0.7);
    }

    .nav {
        display: none !important;
    }

    .nav span {
        padding: 11px 16px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.70);
        border: 1px solid rgba(35, 150, 255, 0.16);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
    }

    .nav span:first-child {
        color: #fff;
        background: linear-gradient(135deg, var(--blue-700), var(--blue-500));
        box-shadow: 0 12px 26px rgba(22, 119, 255, 0.24);
    }

    .top-tagline {
        display: none !important;
    }

    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
        gap: 22px;
        align-items: stretch;
        margin: 6px 0 30px;
    }

    .hero-card,
    .stat-card,
    .result-card,
    .answer-card,
    .info-card {
        border: 1px solid var(--line);
        border-radius: 22px;
        background: var(--glass);
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        padding: 34px 36px;
        color: #fff;
        background:
            radial-gradient(circle at 86% 18%, rgba(34, 211, 238, 0.55), transparent 28%),
            radial-gradient(circle at 18% 88%, rgba(64, 147, 255, 0.34), transparent 34%),
            linear-gradient(135deg, #081a45 0%, #0c3a9a 48%, #147dff 100%);
        box-shadow: 0 28px 80px rgba(8, 36, 92, 0.30);
    }

    .hero-card::before {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: -75px;
        top: -92px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.35);
        background: rgba(255,255,255,0.10);
    }

    .hero-card::after {
        content: "";
        position: absolute;
        inset: auto 34px 24px auto;
        width: 160px;
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(255,255,255,0.25), rgba(34,211,238,0.9));
    }

    .hero-title {
        position: relative;
        font-size: 40px;
        line-height: 1.18;
        font-weight: 950;
        margin: 0 0 16px;
        color: #fff;
    }

    .hero-copy {
        position: relative;
        max-width: 780px;
        font-size: 16px;
        line-height: 1.85;
        color: rgba(255, 255, 255, 0.88);
        margin: 0;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 14px;
        height: 100%;
    }

    .stat-card {
        position: relative;
        overflow: hidden;
        padding: 18px 20px;
        background: rgba(255,255,255,0.86);
    }

    .stat-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 5px;
        background: linear-gradient(180deg, var(--blue-600), var(--cyan-400));
    }

    .stat-label {
        color: var(--muted);
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .stat-value {
        font-size: 27px;
        line-height: 1.1;
        font-weight: 950;
        color: var(--blue-900);
    }

    .section-title {
        position: relative;
        font-size: 25px;
        line-height: 1.15;
        font-weight: 950;
        margin: 0 0 17px;
        color: var(--ink);
    }

    .section-title::after {
        content: "";
        display: block;
        width: 42px;
        height: 4px;
        margin-top: 10px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--blue-600), var(--cyan-400));
    }

    .small-guide {
        padding: 16px 18px;
        background: linear-gradient(135deg, rgba(22, 119, 255, 0.12), rgba(34, 211, 238, 0.10));
        border: 1px solid rgba(35, 150, 255, 0.18);
        border-radius: 18px;
        color: var(--blue-800);
        line-height: 1.65;
        margin: 16px 0 18px;
        font-weight: 850;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 10px 0 20px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 10px 14px;
        border: 1px solid rgba(35, 150, 255, 0.25);
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(232,244,255,0.92));
        color: var(--blue-800);
        font-weight: 950;
        font-size: 13px;
        box-shadow: 0 10px 24px rgba(22,119,255,0.08);
    }

    .chip small {
        color: #5d6b86;
        font-weight: 800;
    }

    .result-card {
        position: relative;
        overflow: hidden;
        padding: 26px 28px;
        border-left: 0;
        background:
            linear-gradient(white, white) padding-box,
            linear-gradient(135deg, var(--blue-600), var(--cyan-400)) border-box;
        border: 1px solid transparent;
        box-shadow: var(--glow);
        margin: 0 0 18px;
    }

    .result-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 7px;
        background: linear-gradient(180deg, var(--blue-600), var(--cyan-400));
    }

    .result-eyebrow {
        color: var(--blue-700);
        font-size: 13px;
        font-weight: 900;
        margin-bottom: 9px;
    }

    .result-title {
        font-size: 35px;
        line-height: 1.16;
        font-weight: 950;
        color: var(--ink);
        margin-bottom: 12px;
    }

    .result-desc {
        color: #344054;
        line-height: 1.75;
        margin: 0;
    }

    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 15px;
    }

    .tag {
        padding: 7px 10px;
        border-radius: 999px;
        background: var(--sky-100);
        color: var(--blue-800);
        font-size: 12px;
        font-weight: 900;
        border: 1px solid rgba(35, 150, 255, 0.16);
    }

    .answer-card {
        position: relative;
        overflow: hidden;
        padding: 22px 24px;
        margin-top: 18px;
        line-height: 1.8;
        color: #10233f;
        border-left: 0;
        background:
            radial-gradient(circle at 100% 0%, rgba(34, 211, 238, 0.16), transparent 28%),
            linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,248,255,0.96));
        box-shadow: var(--glow);
    }

    .answer-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 7px;
        background: linear-gradient(180deg, var(--blue-600), var(--cyan-400));
    }

    .info-card {
        min-height: 158px;
        padding: 21px;
        background: rgba(255,255,255,0.86);
    }

    .info-title {
        font-size: 18px;
        font-weight: 950;
        color: var(--blue-900);
        margin-bottom: 11px;
    }

    .info-text {
        color: #475467;
        line-height: 1.72;
    }

    div[data-testid="stFileUploader"] section {
        border-radius: 18px;
        background: rgba(255,255,255,0.86);
        border: 1px dashed rgba(22, 119, 255, 0.35);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }

    div[data-testid="stFileUploader"] button {
        border-radius: 14px !important;
        border: 1px solid rgba(22,119,255,0.28) !important;
        background: #fff !important;
        color: var(--blue-800) !important;
        font-weight: 900 !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        min-height: 48px;
        border-radius: 16px !important;
        border: 0 !important;
        color: #fff !important;
        font-weight: 950 !important;
        background:
            linear-gradient(135deg, #0b5ed7 0%, #1677ff 52%, #20c7e8 100%) !important;
        box-shadow: 0 16px 36px rgba(22, 119, 255, 0.28), inset 0 1px 0 rgba(255,255,255,0.22) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        filter: saturate(1.1);
        box-shadow: 0 22px 48px rgba(22, 119, 255, 0.36), 0 0 0 4px rgba(35,150,255,0.10) !important;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 15px;
        border: 1px solid rgba(44, 104, 196, 0.18);
        background: rgba(255,255,255,0.92);
        min-height: 46px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85);
    }

    .stAlert {
        border-radius: 16px;
        border: 1px solid rgba(35, 150, 255, 0.14);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(44,104,196,0.16);
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 16px;
        border-radius: 999px 999px 0 0;
        font-weight: 900;
    }

    .stTabs [aria-selected="true"] {
        color: var(--blue-700);
        background: rgba(234, 246, 255, 0.82);
    }

    img {
        border-radius: 18px;
    }

    @media (max-width: 980px) {
        .block-container {
            padding-top: 3.8rem !important;
        }

        .app-header {
            grid-template-columns: 1fr;
            gap: 14px;
        }

        .nav {
            justify-content: flex-start;
            flex-wrap: wrap;
        }

        .hero {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 32px;
        }
    }
</style>
"""


CATEGORY_META = {
    "almirah": {
        "ko": "수납장",
        "symbol": "▤",
        "tags": ["정리", "수납", "생활용품"],
    },
    "chair": {
        "ko": "의자",
        "symbol": "▥",
        "tags": ["좌석", "책상", "작업공간"],
    },
    "fridge": {
        "ko": "냉장고",
        "symbol": "▦",
        "tags": ["주방", "가전", "식품보관"],
    },
    "table": {
        "ko": "테이블",
        "symbol": "▭",
        "tags": ["식탁", "거실", "작업대"],
    },
    "tv": {
        "ko": "TV",
        "symbol": "▣",
        "tags": ["거실", "가전", "시청"],
    },
}


@st.cache_resource
def load_predictor():
    return FurnitureImagePredictor()


@st.cache_resource
def load_rag():
    return FurnitureRAG(KNOWLEDGE_PATH)


def category_en_of(prediction):
    return prediction.get("class_en") or prediction.get("category_en") or ""


def category_ko_of(prediction):
    return prediction.get("class_ko") or prediction.get("category_ko") or ""


def render_header():
    st.markdown(
        """
        <header class="app-header">
            <div class="brand">가구 추천 AI</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_intro(document_count):
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-card">
                <div class="hero-title">이미지 기반 가구 추천 챗봇</div>
                <p class="hero-copy">
                    사용자가 업로드한 가구 이미지를 CNN 모델로 분석하고,
                    RAG 문서 검색을 통해 배치, 스타일, 사용 상황에 맞는 추천 답변을 제공합니다.
                    쇼핑몰의 이미지 검색과 상담 챗봇을 연결하는 프로토타입입니다.
                </p>
            </div>
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-label">분류 모델</div>
                    <div class="stat-value">CNN</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">가구 클래스</div>
                    <div class="stat-value">5개</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">RAG 문서</div>
                    <div class="stat-value">{document_count}개</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_category_chips():
    chips = []
    for category_en, meta in CATEGORY_META.items():
        chips.append(
            (
                '<span class="chip">'
                f'{escape(meta["ko"])} '
                f'<small>{escape(category_en)}</small>'
                "</span>"
            )
        )
    st.markdown(f'<div class="chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_prediction_result(prediction):
    category_en = category_en_of(prediction)
    category_ko = category_ko_of(prediction)
    confidence = prediction["confidence"]
    meta = CATEGORY_META.get(category_en, {})
    symbol = ""
    tags = meta.get("tags", [])

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-eyebrow">현재 이미지 기준 예측 결과</div>
            <div class="result-title">
                {escape(category_ko)}
                <span style="font-size:18px;">({escape(category_en)})</span>
            </div>
            <p class="result-desc">
                모델이 이 이미지를 <b>{escape(category_ko)}</b> 카테고리로 분류했습니다.
                예측 확률은 <b>{confidence * 100:.2f}%</b>입니다.
            </p>
            <div class="tag-row">
                {"".join(f'<span class="tag">{escape(tag)}</span>' for tag in tags)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(probabilities):
    prob_df = pd.DataFrame(probabilities).copy()
    prob_df["percent"] = prob_df["probability"] * 100
    max_probability = prob_df["probability"].max()
    chart = (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusTopRight=10, cornerRadiusBottomRight=10)
        .encode(
            y=alt.Y(
                "category_ko:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelAngle=0, labelLimit=120),
            ),
            x=alt.X(
                "percent:Q",
                title="예측 확률(%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.condition(
                alt.datum.probability == max_probability,
                alt.value("#1677ff"),
                alt.value("#b8dcff"),
            ),
            tooltip=[
                alt.Tooltip("category_ko:N", title="카테고리"),
                alt.Tooltip("probability:Q", title="예측 확률", format=".2%"),
            ],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, use_container_width=True)



predictor = load_predictor()
rag = load_rag()

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_answer_source" not in st.session_state:
    st.session_state.last_answer_source = None
if "last_answer_model" not in st.session_state:
    st.session_state.last_answer_model = None
if "last_answer_error" not in st.session_state:
    st.session_state.last_answer_error = None
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

st.markdown(APP_CSS, unsafe_allow_html=True)
render_header()
render_intro(len(rag.documents))

left_col, right_col = st.columns([0.9, 1.1], gap="large")

with left_col:
    st.markdown('<div class="section-title">가구 이미지 업로드</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "분석할 가구 이미지를 선택하세요.",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    analyze_clicked = st.button("이미지 분석하기", type="primary", use_container_width=True)

    st.markdown(
        '<div class="small-guide">침대, 의자, 수납장, 테이블, TV처럼 가구가 잘 보이는 이미지를 올려주세요.</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="업로드 이미지", use_container_width=True)

    st.markdown(
        '<div class="section-title" style="font-size:20px; margin-top:18px;">분류 가능 카테고리</div>',
        unsafe_allow_html=True,
    )
    render_category_chips()

with right_col:
    st.markdown(
        '<div class="section-title">예측 결과와 추천 상담</div>',
        unsafe_allow_html=True,
    )

    if analyze_clicked:
        if uploaded_file is None:
            st.warning("먼저 이미지를 업로드해주세요.")
        else:
            st.session_state.last_prediction = predictor.predict(uploaded_file)
            st.session_state.last_answer = None
            st.session_state.last_answer_source = None
            st.session_state.last_answer_model = None
            st.session_state.last_answer_error = None
            st.session_state.last_docs = []

    current_prediction = st.session_state.last_prediction

    if current_prediction:
        render_prediction_result(current_prediction)
        render_probability_chart(current_prediction["probabilities"])
    else:
        st.info("이미지를 분석하면 이 영역에 예측 카테고리와 확률이 표시됩니다.")

    with st.form("recommend_form"):
        question = st.text_input(
            "추천 질문",
            placeholder="예: 이 가구를 작은 원룸에 어울리게 배치하려면 어떻게 하면 좋을까?",
        )
        recommend_clicked = st.form_submit_button("추천 답변 받기", use_container_width=True)

    if recommend_clicked:
        category_en = None
        category_ko = None
        confidence = None

        if current_prediction:
            category_en = category_en_of(current_prediction)
            category_ko = category_ko_of(current_prediction)
            confidence = current_prediction["confidence"]

        query = question.strip() or "공간에 어울리는 가구 추천"
        docs = rag.retrieve(
            query=f"{category_ko or ''} {category_en or ''} {query}",
            category=category_en,
            top_k=1,
        )
        st.session_state.last_docs = docs
        with st.spinner("Generating recommendation answer..."):
            answer_result = rag.generate_chat_answer_with_meta(
                question=query,
                retrieved_docs=docs,
                category_ko=category_ko,
                category_en=category_en,
                confidence=confidence,
            )
        st.session_state.last_answer = answer_result["answer"]
        st.session_state.last_answer_source = answer_result["source"]
        st.session_state.last_answer_model = answer_result["model"]
        st.session_state.last_answer_error = answer_result["error"]

    if st.session_state.last_answer:
        answer_html = escape(st.session_state.last_answer).replace("\n", "<br>")
        source_label_map = {
            "gemini": "Gemini API",
            "groq": "Groq API",
            "fallback": "RAG 문서 기반 답변",
        }
        source_key = st.session_state.last_answer_source or "fallback"
        source_label = source_label_map.get(source_key, "RAG document answer")
        model_label = st.session_state.last_answer_model or source_label
        source_html = escape(f"{source_label} / {model_label}")
        st.markdown(
            f"""
            <div class="answer-card">
                <div style="font-size:13px; font-weight:700; color:#1f6fd1; margin-bottom:10px;">{source_html}</div>
                <b>추천 답변</b><br><br>
                {answer_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.last_answer_source == "fallback":
            st.info("외부 AI 응답이 없어 RAG 문서 기반 답변으로 대체했습니다.")

        with st.expander("검색된 RAG 근거 문서 보기"):
            for doc in st.session_state.last_docs:
                st.write(f"**{doc.title}**")
                st.write(doc.content)

st.divider()
info_tab, data_tab = st.tabs(["프로젝트 정보", "RAG 데이터"])

with info_tab:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">이미지 분류</div>
                <div class="info-text">
                    업로드 이미지를 160x160 RGB 형태로 변환하고 CNN 기반 모델로 가구 카테고리를 예측합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">RAG 검색</div>
                <div class="info-text">
                    예측 카테고리와 사용자 질문을 조합해 관련 추천 문서를 검색하고 답변 근거로 사용합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">확장 방향</div>
                <div class="info-text">
                    실제 상품 DB, 이미지 검색, LLM API를 연결하면 쇼핑몰 추천 기능으로 확장할 수 있습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with data_tab:
    st.write("현재 프로토타입은 TF-IDF 기반 검색과 템플릿 답변으로 RAG 흐름을 확인하는 구조입니다.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "title": doc.title,
                    "category": doc.category,
                    "content": doc.content,
                }
                for doc in rag.documents
            ]
        ),
        use_container_width=True,
    )
