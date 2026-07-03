from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

from image_predictor import FurnitureImagePredictor
from rag_core import FurnitureRAG


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "data" / "furniture_knowledge.md"


st.set_page_config(
    page_title="RAG 기반 가구 추천 챗봇",
    page_icon="",
    layout="wide",
)


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
            "content": "가구 이미지나 질문을 바탕으로 추천 정보를 도와줄게요. 궁금한 내용을 입력해주세요.",
        }
    ]


predictor = load_predictor()
rag = load_rag()

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "chat_messages" not in st.session_state:
    reset_chat()

st.title("딥러닝 이미지 분류 + RAG 기반 가구 추천")
st.caption("전이학습 이미지 분류 모델과 검색 기반 추천 문서를 결합한 LLM/RAG 프로젝트 프로토타입")

tabs = st.tabs(["프로젝트 설명", "추천 챗봇", "RAG 데이터"])

with tabs[0]:
    st.header("프로젝트 설명")
    st.write(
        """
        사용자가 가구 이미지를 업로드하면 Kaggle 환경에서 전이학습한 이미지 분류 모델이
        가구 카테고리를 예측합니다. 이후 예측된 카테고리와 사용자의 질문을 기준으로
        가구 추천 문서를 검색하고, 검색된 근거를 바탕으로 추천 답변을 생성합니다.

        현재 버전은 외부 LLM API 없이 RAG 검색 흐름을 확인할 수 있는 1차 프로토타입입니다.
        이후 LangChain, Vector Store, LLM API를 연결하면 실제 LLM 답변 생성 구조로 확장할 수 있습니다.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("이미지 분류 모델", "MobileNetV2")
    c2.metric("RAG 지식 데이터", f"{len(rag.documents)}개 청크")
    c3.metric("분류 클래스", "5개")

    st.subheader("전체 처리 흐름")
    st.write(
        """
        1. 사용자가 가구 이미지를 업로드한다.  
        2. 이미지 분류 모델이 이미지를 수납장, 의자, 냉장고, 테이블, TV 중 하나로 분류한다.  
        3. 예측 카테고리와 사용자 질문을 조합하여 관련 추천 문서를 검색한다.  
        4. 검색된 근거를 바탕으로 추천 답변을 생성한다.  
        """
    )

with tabs[1]:
    st.header("이미지 기반 가구 추천 챗봇")
    st.write(
        """
        이미지를 먼저 분석하면 챗봇이 예측된 가구 카테고리를 참고해서 답변합니다.
        이미지를 올리지 않아도 일반적인 가구 추천 질문을 할 수 있습니다.
        """
    )

    image_col, result_col = st.columns([0.9, 1.1])

    with image_col:
        uploaded_file = st.file_uploader(
            "가구 이미지를 업로드하세요",
            type=["jpg", "jpeg", "png"],
        )
        analyze_clicked = st.button("이미지 분석하기", type="primary", use_container_width=True)

        if uploaded_file is not None:
            st.image(uploaded_file, caption="업로드 이미지", use_container_width=True)

    with result_col:
        st.subheader("이미지 분석 결과")

        if analyze_clicked:
            if uploaded_file is None:
                st.warning("먼저 이미지를 업로드해주세요.")
            else:
                prediction = predictor.predict(uploaded_file)
                category_en = prediction["class_en"]
                category_ko = prediction["class_ko"]
                confidence = prediction["confidence"]
                st.session_state.last_prediction = {
                    "category_en": category_en,
                    "category_ko": category_ko,
                    "confidence": confidence,
                    "original_size": prediction.get("original_size"),
                    "preprocess_mode": prediction.get("preprocess_mode"),
                    "preprocess_count": prediction.get("preprocess_count"),
                    "probabilities": prediction["probabilities"],
                    "image_size": prediction.get("image_size"),
                }

        current_prediction = st.session_state.last_prediction

        if current_prediction:
            st.success(
                f"현재 이미지 컨텍스트: {current_prediction['category_ko']} "
                f"({current_prediction['category_en']})"
            )
            st.write(f"예측 확률: **{current_prediction['confidence'] * 100:.2f}%**")
            if current_prediction.get("original_size"):
                width, height = current_prediction["original_size"]
                image_size = current_prediction.get("image_size", (160, 160))
                st.caption(
                    f"원본 이미지: {width} x {height}px / "
                    f"전처리: {image_size[0]} x {image_size[1]} 변환 "
                    f"{current_prediction.get('preprocess_count', 1)}개 평균 예측"
                )

            if current_prediction["confidence"] < 0.6:
                st.warning(
                    "예측 확률이 높지 않습니다. 업로드 이미지의 각도, 배경, 조명에 따라 "
                    "결과가 달라질 수 있으므로 추천을 시작하기 위한 참고값으로 해석하는 것이 좋습니다."
                )

            prob_df = pd.DataFrame(current_prediction["probabilities"])
            chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("category_ko:N", title="카테고리", sort=None),
                    y=alt.Y("probability:Q", title="예측 확률", scale=alt.Scale(domain=[0, 1])),
                    tooltip=[
                        alt.Tooltip("category_ko:N", title="카테고리"),
                        alt.Tooltip("probability:Q", title="예측 확률", format=".2%"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("이미지를 분석하면 이 영역에 예측 카테고리와 확률이 표시됩니다.")

    st.divider()
    st.subheader("RAG 추천 대화")

    chat_left, chat_right = st.columns([1.1, 0.9])

    with chat_left:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        prompt = st.chat_input("예: 이 가구는 어떤 공간에 어울려?")

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            current_prediction = st.session_state.last_prediction
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

    with chat_right:
        st.write("현재 RAG 기준")
        if st.session_state.last_prediction:
            st.info(
                f"이미지 카테고리: {st.session_state.last_prediction['category_ko']} "
                f"({st.session_state.last_prediction['category_en']})"
            )
        else:
            st.info("이미지 카테고리 없이 질문 내용만으로 문서를 검색합니다.")

        if st.button("대화 초기화", use_container_width=True):
            reset_chat()
            st.rerun()

        st.caption("검색 대상 문서")
        st.write("수납장, 의자, 냉장고, 테이블, TV 추천 가이드와 배치 팁")

with tabs[2]:
    st.header("RAG 지식 데이터")
    st.write(
        """
        RAG는 사용자의 질문에 바로 답변하지 않고, 먼저 관련 문서를 검색한 뒤
        검색 결과를 바탕으로 답변을 생성하는 방식입니다.
        """
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
