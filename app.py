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
    page_title="PlusHome AI 추천",
    page_icon="🏠",
    layout="wide",
)


PLUS_HOME_CSS = """
<style>
    :root {
        --plus-blue: #1976d2;
        --plus-blue-dark: #0f5fb8;
        --plus-red: #e53935;
        --plus-text: #111827;
        --plus-muted: #6b7280;
        --plus-line: #e5e7eb;
        --plus-soft: #f7f9fc;
    }

    .stApp {
        background: #ffffff;
        color: var(--plus-text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--plus-text);
    }

    div[data-testid="stTabs"] button {
        font-weight: 700;
        color: #111827;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--plus-blue);
        border-bottom-color: var(--plus-blue);
    }

    .plus-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 10px 0 18px;
        border-bottom: 1px solid var(--plus-line);
        margin-bottom: 22px;
    }

    .plus-logo {
        font-size: 28px;
        font-weight: 900;
        color: #0b0f19;
        white-space: nowrap;
    }

    .plus-nav {
        display: flex;
        align-items: center;
        gap: 28px;
        font-weight: 800;
        color: #111827;
        white-space: nowrap;
    }

    .plus-search {
        flex: 1;
        min-width: 240px;
        max-width: 420px;
        height: 42px;
        display: flex;
        align-items: center;
        padding: 0 14px;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        color: #9ca3af;
        font-size: 14px;
        background: white;
    }

    .plus-user {
        color: var(--plus-blue);
        font-weight: 800;
        white-space: nowrap;
    }

    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.8fr);
        gap: 28px;
        align-items: stretch;
        margin: 10px 0 26px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 900;
        margin: 0 0 10px;
        color: #111827;
    }

    .hero-copy {
        font-size: 16px;
        line-height: 1.75;
        color: #374151;
        margin: 0;
    }

    .hero-card,
    .panel-card,
    .result-card,
    .mini-card {
        border: 1px solid var(--plus-line);
        border-radius: 8px;
        background: white;
    }

    .hero-card {
        padding: 22px 24px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
    }

    .summary-item {
        padding: 16px;
        border: 1px solid var(--plus-line);
        border-radius: 8px;
        background: var(--plus-soft);
    }

    .summary-label {
        color: var(--plus-muted);
        font-size: 13px;
        margin-bottom: 8px;
    }

    .summary-value {
        font-size: 24px;
        font-weight: 900;
        color: #111827;
    }

    .panel-card {
        padding: 20px;
        margin-bottom: 18px;
    }

    .panel-title {
        font-size: 20px;
        font-weight: 900;
        margin: 0 0 12px;
    }

    .result-card {
        padding: 22px;
        border-left: 6px solid var(--plus-blue);
        background: #fbfdff;
        margin-bottom: 16px;
    }

    .result-eyebrow {
        color: var(--plus-muted);
        font-size: 13px;
        margin-bottom: 8px;
    }

    .result-title {
        font-size: 34px;
        line-height: 1.15;
        font-weight: 900;
        color: #111827;
        margin-bottom: 8px;
    }

    .result-desc {
        color: #374151;
        line-height: 1.7;
        margin: 0;
    }

    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
    }

    .tag {
        display: inline-flex;
        align-items: center;
        padding: 5px 10px;
        border-radius: 999px;
        border: 1px solid var(--plus-blue);
        color: var(--plus-blue);
        background: #ffffff;
        font-size: 13px;
        font-weight: 800;
    }

    .mini-card {
        padding: 16px 18px;
        min-height: 112px;
    }

    .mini-card-title {
        color: var(--plus-muted);
        font-size: 13px;
        margin-bottom: 10px;
    }

    .mini-card-value {
        font-size: 20px;
        font-weight: 900;
    }

    .product-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
    }

    .product-card {
        border: 1px solid #dcdfe5;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        background: white;
        min-height: 148px;
    }

    .product-icon {
        font-size: 34px;
        margin-bottom: 8px;
    }

    .product-name {
        font-weight: 900;
        margin-bottom: 6px;
    }

    .product-caption {
        font-size: 13px;
        color: var(--plus-muted);
        line-height: 1.45;
    }

    .section-line {
        border-top: 1px solid var(--plus-line);
        margin: 20px 0;
    }

    .blue-note {
        border-left: 5px solid var(--plus-blue);
        background: #eef6ff;
        padding: 14px 16px;
        border-radius: 6px;
        color: #0f3765;
        line-height: 1.65;
        margin: 12px 0;
    }

    .stButton > button {
        border-radius: 6px;
        font-weight: 800;
        min-height: 42px;
    }

    .stButton > button[kind="primary"] {
        background: var(--plus-blue);
        border-color: var(--plus-blue);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--plus-blue-dark);
        border-color: var(--plus-blue-dark);
    }

    @media (max-width: 900px) {
        .plus-header,
        .hero {
            grid-template-columns: 1fr;
            display: block;
        }

        .plus-header {
            display: flex;
            flex-wrap: wrap;
        }

        .summary-grid,
        .product-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


CATEGORY_META = {
    "almirah": {
        "ko": "수납장",
        "icon": "🗄️",
        "tags": ["수납", "정리", "벽면 배치"],
        "copy": "생활용품과 문서를 정리하기 좋은 보관형 가구입니다.",
    },
    "chair": {
        "ko": "의자",
        "icon": "🪑",
        "tags": ["착석감", "책상", "포인트"],
        "copy": "작업 공간이나 식탁 주변에서 사용성이 중요한 가구입니다.",
    },
    "fridge": {
        "ko": "냉장고",
        "icon": "🧊",
        "tags": ["주방", "용량", "동선"],
        "copy": "주방 동선과 설치 공간을 함께 고려해야 하는 대형 가전입니다.",
    },
    "table": {
        "ko": "테이블",
        "icon": "🪵",
        "tags": ["식사", "작업", "거실"],
        "copy": "식사, 작업, 수납 등 사용 목적에 따라 선택 기준이 달라집니다.",
    },
    "tv": {
        "ko": "TV",
        "icon": "📺",
        "tags": ["시청거리", "거실", "배치"],
        "copy": "시청 거리와 빛 반사를 고려해 배치해야 하는 가전입니다.",
    },
}


@st.cache_resource
def load_predictor():
    return FurnitureImagePredictor()


@st.cache_resource
def load_rag():
    return FurnitureRAG(KNOWLEDGE_PATH)


def reset_chat():
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "가구 이미지를 먼저 분석하면 예측된 카테고리를 기준으로 추천해드릴게요. "
                "이미지 없이도 원하는 공간, 스타일, 예산을 말해주면 관련 문서를 찾아 답변할 수 있어요."
            ),
        }
    ]


def render_header():
    st.markdown(
        """
        <div class="plus-header">
            <div class="plus-logo">PlusHome AI</div>
            <div class="plus-nav">
                <span>쇼핑</span>
                <span>인테리어</span>
                <span>추천</span>
            </div>
            <div class="plus-search">이미지로 가구를 찾고 추천받아보세요</div>
            <div class="plus-user">momo님</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_intro(rag_document_count):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-card">
                <div class="hero-title">이미지 기반 가구 추천 챗봇</div>
                <p class="hero-copy">
                    PlusHome의 쇼핑·인테리어 흐름에 맞춰, 사용자가 업로드한 가구 이미지를 분석하고
                    RAG 문서 검색을 통해 배치, 스타일, 사용 상황에 맞는 추천 답변을 제공합니다.
                </p>
            </div>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">분류 모델</div>
                    <div class="summary-value">CNN</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">가구 클래스</div>
                    <div class="summary-value">5개</div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">RAG 문서</div>
                    <div class="summary-value">{rag_document_count}개</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_category_cards():
    cards = []
    for category_en, meta in CATEGORY_META.items():
        cards.append(
            f"""
            <div class="product-card">
                <div class="product-icon">{meta["icon"]}</div>
                <div class="product-name">{meta["ko"]}</div>
                <div class="product-caption">{meta["copy"]}</div>
            </div>
            """
        )
    st.markdown(f'<div class="product-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_prediction_result(prediction):
    category_en = prediction["category_en"]
    category_ko = prediction["category_ko"]
    confidence = prediction["confidence"]
    meta = CATEGORY_META.get(category_en, {})
    tags = meta.get("tags", [])
    icon = meta.get("icon", "🛋️")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-eyebrow">현재 이미지 기준 예측 결과</div>
            <div class="result-title">{icon} {escape(category_ko)} <span style="font-size:20px;">({escape(category_en)})</span></div>
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

    size_text = "-"
    if prediction.get("original_size"):
        width, height = prediction["original_size"]
        image_size = prediction.get("image_size", (160, 160))
        size_text = f"{width}x{height}px → {image_size[0]}x{image_size[1]}px"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-card-title">예측 확률</div>
                <div class="mini-card-value">{confidence * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-card-title">이미지 처리</div>
                <div class="mini-card-value">{prediction.get("preprocess_count", 1)}회 평균</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-card-title">입력 크기</div>
                <div class="mini-card-value" style="font-size:16px;">{escape(size_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if confidence < 0.6:
        st.warning(
            "예측 확률이 높지 않습니다. 배경, 조명, 촬영 각도에 따라 결과가 달라질 수 있으니 추천을 위한 참고값으로 봐주세요."
        )


def render_probability_chart(probabilities):
    prob_df = pd.DataFrame(probabilities).copy()
    chart = (
        alt.Chart(prob_df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("category_ko:N", title="카테고리", sort=None),
            y=alt.Y("probability:Q", title="예측 확률", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(
                alt.datum.probability == prob_df["probability"].max(),
                alt.value("#1976D2"),
                alt.value("#BFD7F2"),
            ),
            tooltip=[
                alt.Tooltip("category_ko:N", title="카테고리"),
                alt.Tooltip("probability:Q", title="예측 확률", format=".2%"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


predictor = load_predictor()
rag = load_rag()

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "chat_messages" not in st.session_state:
    reset_chat()

st.markdown(PLUS_HOME_CSS, unsafe_allow_html=True)
render_header()
render_intro(len(rag.documents))

tabs = st.tabs(["이미지 추천", "추천 챗봇", "프로젝트 정보", "RAG 데이터"])

with tabs[0]:
    left_col, right_col = st.columns([0.95, 1.05])

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">가구 이미지 업로드</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "분석할 가구 이미지를 선택하세요",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("이미지 분석하기", type="primary", use_container_width=True)
        if uploaded_file is not None:
            st.image(uploaded_file, caption="업로드 이미지", use_container_width=True)
        else:
            st.info("침대, 의자, 수납장, 테이블, TV처럼 가구가 잘 보이는 이미지를 올려주세요.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">분류 가능 카테고리</div>', unsafe_allow_html=True)
        render_category_cards()
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">예측 결과</div>', unsafe_allow_html=True)

        if analyze_clicked:
            if uploaded_file is None:
                st.warning("먼저 이미지를 업로드해주세요.")
            else:
                st.session_state.last_prediction = predictor.predict(uploaded_file)

        current_prediction = st.session_state.last_prediction
        if current_prediction:
            render_prediction_result(current_prediction)
            st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">클래스별 예측 확률</div>', unsafe_allow_html=True)
            render_probability_chart(current_prediction["probabilities"])
        else:
            st.info("이미지를 분석하면 이 영역에 예측 카테고리와 확률이 표시됩니다.")
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">PlusHome AI 추천 상담</div>', unsafe_allow_html=True)

    current_prediction = st.session_state.last_prediction
    if current_prediction:
        st.markdown(
            f"""
            <div class="blue-note">
                현재 이미지 분석 결과는 <b>{escape(current_prediction["category_ko"])}</b>입니다.
                이 카테고리를 우선 참고해 추천 답변을 생성합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="blue-note">
                이미지를 먼저 분석하면 추천 답변이 더 구체적입니다.
                이미지 없이도 공간, 스타일, 예산을 입력하면 RAG 문서를 기준으로 답변할 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    chat_col, guide_col = st.columns([1.35, 0.65])

    with chat_col:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("예: 좁은 원룸에 어울리게 배치하려면 어떻게 하면 좋을까?")

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            category_en = None
            category_ko = None
            confidence = None

            if current_prediction:
                category_en = current_prediction["category_en"]
                category_ko = current_prediction["category_ko"]
                confidence = current_prediction["confidence"]

            docs = rag.retrieve(
                query=f"{category_ko or ''} {category_en or ''} {prompt}",
                category=category_en,
                top_k=3,
            )
            answer = rag.generate_chat_answer(
                question=prompt,
                retrieved_docs=docs,
                category_ko=category_ko,
                category_en=category_en,
                confidence=confidence,
            )
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()

    with guide_col:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-card-title">추천 질문 예시</div>', unsafe_allow_html=True)
        st.write("- 이 가구는 어느 공간에 어울려?")
        st.write("- 작은 방에 배치할 때 주의할 점은?")
        st.write("- 모던한 분위기로 맞추려면?")
        st.write("- 비슷한 상품을 추천한다면?")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("대화 초기화", use_container_width=True):
            reset_chat()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">프로젝트 설명</div>', unsafe_allow_html=True)
    st.write(
        """
        이 프로젝트는 딥러닝 이미지 분류 모델과 RAG 검색 흐름을 결합한 가구 추천 프로토타입입니다.
        사용자가 이미지를 업로드하면 CNN 기반 모델이 가구 카테고리를 예측하고, RAG는 예측 카테고리와 질문을 바탕으로
        관련 추천 문서를 검색해 답변 근거로 활용합니다.
        """
    )
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("이미지 입력 크기", "160 x 160")
    c2.metric("분류 대상", "5 classes")
    c3.metric("추천 방식", "RAG 검색")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">처리 흐름</div>', unsafe_allow_html=True)
    st.write("1. 이미지 업로드")
    st.write("2. CNN 모델로 가구 카테고리 예측")
    st.write("3. 예측 카테고리와 사용자 질문을 조합해 문서 검색")
    st.write("4. 검색된 문서를 근거로 추천 답변 생성")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">RAG 지식 데이터</div>', unsafe_allow_html=True)
    st.write(
        "현재 프로토타입은 외부 LLM API 없이 TF-IDF 기반 검색과 템플릿 답변으로 RAG 흐름을 확인하는 구조입니다."
    )
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
    st.markdown("</div>", unsafe_allow_html=True)
