# RAG 기반 가구 추천 챗봇

딥러닝 이미지 분류 모델과 RAG 검색 구조를 결합한 LLM 프로젝트 프로토타입입니다.

## 프로젝트 흐름

1. 사용자가 가구 이미지를 업로드합니다.
2. Kaggle에서 전이학습한 이미지 분류 모델이 이미지를 `almirah`, `chair`, `fridge`, `table`, `tv` 중 하나로 분류합니다.
3. 예측된 카테고리와 사용자 질문을 바탕으로 가구 추천 지식 문서를 검색합니다.
4. 검색된 근거를 기반으로 추천 답변을 생성합니다.
5. 챗봇 화면에서 추가 질문을 입력하면 이전 이미지 예측 결과를 참고하여 대화형 답변을 제공합니다.
   Gemini 또는 Groq API 키가 설정되어 있으면 외부 AI 응답을 사용하고, 연결이 실패하면 RAG 문서 기반 답변으로 대체합니다.

## 주요 파일

- `app.py`: Streamlit 시연 앱
- `image_predictor.py`: 이미지 전처리 및 모델 예측 코드
- `rag_core.py`: TF-IDF 기반 RAG 검색 및 답변 생성 코드
- `data/furniture_knowledge.md`: RAG 검색용 가구 추천 문서
- `furniture_transfer_160_compat.keras`: Streamlit 실행용 호환 모델
- `class_names.json`: 모델 클래스명 및 입력 이미지 크기 정보

## 실행 방법

```bash
cd C:\AIProject\llm_mohayoung_project
C:\AIProject\.venv\Scripts\streamlit run app.py
```

## 현재 버전

- Gemini/Groq API 키가 있으면 외부 LLM 응답을 생성하고, 키가 없거나 API 연결이 실패하면 RAG 검색 흐름을 확인할 수 있는 대체 답변을 제공합니다.
- 이미지 분류는 160x160 RGB 입력 기반 전이학습 모델을 사용합니다.
- 이후 LangChain, Vector Store, LLM API를 연결하면 실제 LLM 응답 생성 구조로 확장할 수 있습니다.
