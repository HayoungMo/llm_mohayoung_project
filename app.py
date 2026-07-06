from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from image_predictor import FurnitureImagePredictor
from rag_core import FurnitureRAG


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "data" / "furniture_knowledge.md"


st.set_page_config(
    page_title="가구 추천 AI",
    page_icon="home",
    layout="wide",
)


APP_CSS = """
<style>
    :root {
        --brand-blue: #1976d2;
        --brand-blue-dark: #0f5fb8;
        --brand-red: #e53935;
        --text-main: #111827;
        --text-muted: #6b7280;
        --line: #e5e7eb;
        --panel: #f8fafc;
        --note: #eaf4ff;
    }

    .stApp {
        background: #ffffff;
        color: var(--text-main);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem !important;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--text-main);
    }

    .app-header {
        display: grid;
        grid-template-columns: 180px minmax(260px, 1fr) auto;
        align-items: center;
        gap: 22px;
        min-height: 62px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 22px;
        overflow: visible;
    }

    .brand {
        font-size: 26px;
        font-weight: 900;
        color: #0b0f19;
        white-space: nowrap;
    }

    .nav {
        display: flex;
        align-items: center;
        gap: 24px;
        font-size: 15px;
        font-weight: 800;
        color: #111827;
        white-space: nowrap;
    }

    .top-tagline {
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 18px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        color: #8b95a1;
        font-size: 14px;
        background: #fff;
        min-width: 260px;
    }

    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(300px, 0.75fr);
        gap: 24px;
        align-items: stretch;
        margin: 4px 0 26px;
    }

    .hero-card,
    .stat-card,
    .result-card,
    .answer-card,
    .info-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: white;
    }

    .hero-card {
        padding: 24px 28px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }

    .hero-title {
        font-size: 34px;
        line-height: 1.22;
        font-weight: 900;
        margin: 0 0 12px;
    }

    .hero-copy {
        font-size: 16px;
        line-height: 1.75;
        color: #374151;
        margin: 0;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        height: 100%;
    }

    .stat-card {
        padding: 18px 16px;
        background: var(--panel);
    }

    .stat-label {
        color: var(--text-muted);
        font-size: 13px;
        margin-bottom: 10px;
    }

    .stat-value {
        font-size: 24px;
        font-weight: 900;
    }

    .section-title {
        font-size: 23px;
        font-weight: 900;
        margin: 0 0 14px;
    }

    .small-guide {
        padding: 13px 15px;
        background: var(--note);
        border-radius: 7px;
        color: #0f5fb8;
        line-height: 1.6;
        margin: 12px 0 16px;
        font-weight: 700;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 18px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 7px 11px;
        border: 1px solid #d7e4f5;
        border-radius: 999px;
        background: #fff;
        color: #0f5fb8;
        font-weight: 800;
        font-size: 13px;
    }

    .chip small {
        color: #6b7280;
        font-weight: 700;
    }

    .result-card {
        padding: 22px;
        border-left: 6px solid var(--brand-blue);
        background: #fbfdff;
        margin: 0 0 16px;
    }

    .result-eyebrow {
        color: var(--text-muted);
        font-size: 13px;
        margin-bottom: 8px;
    }

    .result-title {
        font-size: 32px;
        line-height: 1.18;
        font-weight: 900;
        color: #111827;
        margin-bottom: 10px;
    }

    .result-desc {
        color: #374151;
        line-height: 1.7;
        margin: 0;
    }

    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 14px;
    }

    .tag {
        padding: 5px 9px;
        border-radius: 999px;
        background: #eaf4ff;
        color: #0f5fb8;
        font-size: 12px;
        font-weight: 800;
    }

    .answer-card {
        padding: 18px 20px;
        margin-top: 16px;
        line-height: 1.75;
        border-left: 6px solid var(--brand-red);
        background: #fffaf6;
    }

    .info-card {
        min-height: 148px;
        padding: 18px;
        background: #fff;
    }

    .info-title {
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .info-text {
        color: #4b5563;
        line-height: 1.65;
    }

    div[data-testid="stFileUploader"] section {
        border-radius: 8px;
        background: #f1f5f9;
        border: 1px solid #e5e7eb;
    }

    .stButton > button {
        border-radius: 6px;
        font-weight: 800;
        min-height: 42px;
    }

    @media (max-width: 900px) {
        .app-header {
            grid-template-columns: 1fr;
            gap: 12px;
        }

        .hero {
            grid-template-columns: 1fr;
        }

        .stat-grid {
            grid-template-columns: 1fr;
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
            <nav class="nav">
                <span>이미지 분석</span>
                <span>추천 상담</span>
                <span>RAG 검색</span>
            </nav>
            <div class="top-tagline">이미지로 가구를 분석하고 추천받아보세요</div>
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
                f'{escape(meta["symbol"])} {escape(meta["ko"])} '
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
    symbol = meta.get("symbol", "▣")
    tags = meta.get("tags", [])

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-eyebrow">현재 이미지 기준 예측 결과</div>
            <div class="result-title">
                {escape(symbol)} {escape(category_ko)}
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
    max_probability = prob_df["probability"].max()
    chart = (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("category_ko:N", title="카테고리", sort=None),
            y=alt.Y("probability:Q", title="예측 확률", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(
                alt.datum.probability == max_probability,
                alt.value("#1976D2"),
                alt.value("#BFD7F2"),
            ),
            tooltip=[
                alt.Tooltip("category_ko:N", title="카테고리"),
                alt.Tooltip("probability:Q", title="예측 확률", format=".2%"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


predictor = load_predictor()
rag = load_rag()

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
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
            top_k=3,
        )
        st.session_state.last_docs = docs
        st.session_state.last_answer = rag.generate_chat_answer(
            question=query,
            retrieved_docs=docs,
            category_ko=category_ko,
            category_en=category_en,
            confidence=confidence,
        )

    if st.session_state.last_answer:
        answer_html = escape(st.session_state.last_answer).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="answer-card">
                <b>추천 답변</b><br><br>
                {answer_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

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
