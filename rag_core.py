import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RAGDocument:
    title: str
    category: str
    content: str


class FurnitureRAG:
    def __init__(self, knowledge_path):
        self.knowledge_path = Path(knowledge_path)
        self.documents = self._load_documents()
        self.vectorizer = TfidfVectorizer()
        self.doc_texts = [
            f"{doc.title} {doc.category} {doc.content}" for doc in self.documents
        ]
        self.matrix = self.vectorizer.fit_transform(self.doc_texts)

    def _load_documents(self):
        if not self.knowledge_path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {self.knowledge_path}")

        text = self.knowledge_path.read_text(encoding="utf-8")
        sections = re.split(r"\n## ", text)
        documents = []

        for raw_section in sections:
            section = raw_section.strip()
            if not section:
                continue

            lines = section.splitlines()
            title_line = lines[0].replace("## ", "").strip()
            if title_line.startswith("# "):
                continue
            content = "\n".join(lines[1:]).strip()
            match = re.search(r"\[(.*?)\]", title_line)
            category = match.group(1) if match else "general"

            documents.append(
                RAGDocument(
                    title=title_line,
                    category=category,
                    content=content,
                )
            )

        return documents

    def retrieve(self, query, category=None, top_k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]

        ranked = []
        for idx, score in enumerate(scores):
            doc = self.documents[idx]
            category_bonus = 0.2 if category and doc.category == category else 0
            ranked.append((score + category_bonus, doc))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]

    def generate_answer(self, category_ko, category_en, question, retrieved_docs):
        evidence = "\n".join(f"- {doc.content}" for doc in retrieved_docs)
        return f"""
현재 이미지는 **{category_ko}({category_en})** 카테고리로 예측되었습니다.

질문: {question}

검색된 가구 추천 문서를 기준으로 보면, 이 가구는 공간의 용도와 분위기에 맞춰 배치하는 것이 좋습니다.
특히 아래 근거를 참고할 수 있습니다.

{evidence}

        따라서 이 이미지는 {category_ko} 계열 상품으로 보고, 사용 공간의 크기와 원하는 분위기에 맞춰
소재, 색상, 배치 조합을 함께 고려하는 방식으로 추천할 수 있습니다.
""".strip()

    def generate_chat_answer(
        self,
        question,
        retrieved_docs,
        category_ko=None,
        category_en=None,
        confidence=None,
    ):
        context_text = ""
        if category_ko and category_en:
            context_text = (
                f"현재 이미지 분류 결과는 {category_ko}({category_en})입니다. "
                "이 카테고리를 우선 참고했습니다.\n\n"
            )

        confidence_note = ""
        if confidence is not None and confidence < 0.6:
            confidence_note = (
                "다만 이미지 분류 확률이 아주 높지는 않으므로, 현재 카테고리는 "
                "확정값이 아니라 추천을 시작하기 위한 참고 정보로 보는 것이 좋습니다.\n\n"
            )

        evidence = "\n".join(f"- {doc.content}" for doc in retrieved_docs)
        return f"""
{context_text}{confidence_note}질문: {question}

검색된 가구 추천 문서를 기준으로 답변하면 다음과 같습니다.

{evidence}

정리하면, 질문의 핵심은 공간의 용도와 원하는 분위기에 맞춰 가구의 소재, 색상, 크기, 배치 조합을 함께 고려하는 것입니다.
현재 답변은 RAG 검색 결과를 바탕으로 생성된 프로토타입 답변이며, 이후 LLM API를 연결하면 더 자연스러운 문장으로 확장할 수 있습니다.
""".strip()
