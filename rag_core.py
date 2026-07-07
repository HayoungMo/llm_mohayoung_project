import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

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
        self.ollama_url = self._ollama_generate_url()
        self.ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "240"))
        self.ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "80"))
        self.ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
        self.model_candidates = self._model_candidates()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        gemini_model = os.getenv("GEMINI_MODEL", "").strip()
        self.gemini_models = (
            [gemini_model]
            if gemini_model
            else ["gemini-1.5-flash", "gemini-1.5-flash-latest"]
        )
        self.gemini_url_template = os.getenv(
            "GEMINI_API_URL",
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        ).strip()
        self.gemini_timeout = float(os.getenv("GEMINI_TIMEOUT", "60"))

        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_url = os.getenv(
            "GROQ_API_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        ).strip()
        groq_model = os.getenv("GROQ_MODEL", "").strip()
        self.groq_models = (
            [groq_model]
            if groq_model
            else ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"]
        )
        self.groq_timeout = float(os.getenv("GROQ_TIMEOUT", "60"))

    def _ollama_generate_url(self):
        raw_url = (
            os.getenv("OLLAMA_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).strip().rstrip("/")
        if raw_url.endswith("/api/generate"):
            return raw_url
        if raw_url.endswith("/api"):
            return f"{raw_url}/generate"
        return f"{raw_url}/api/generate"

    def _model_candidates(self):
        env_model = os.getenv("OLLAMA_MODEL")
        candidates = []
        if env_model:
            candidates.append(env_model)
        candidates.extend(["gemma2:latest", "gemma4:e2b"])
        deduped = []
        for model in candidates:
            if model and model not in deduped:
                deduped.append(model)
        return deduped

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

    def generate_chat_answer(
        self,
        question,
        retrieved_docs,
        category_ko=None,
        category_en=None,
        confidence=None,
    ):
        result = self.generate_chat_answer_with_meta(
            question=question,
            retrieved_docs=retrieved_docs,
            category_ko=category_ko,
            category_en=category_en,
            confidence=confidence,
        )
        return result["answer"]

    def generate_chat_answer_with_meta(
        self,
        question,
        retrieved_docs,
        category_ko=None,
        category_en=None,
        confidence=None,
    ):
        fallback_answer = self._template_answer(
            question=question,
            retrieved_docs=retrieved_docs,
            category_ko=category_ko,
            category_en=category_en,
            confidence=confidence,
        )
        prompt = self._build_ollama_prompt(
            question=question,
            retrieved_docs=retrieved_docs,
            category_ko=category_ko,
            category_en=category_en,
            confidence=confidence,
        )

        last_error = None

        if self.gemini_api_key:
            for model in self.gemini_models:
                try:
                    answer = self._call_gemini(model=model, prompt=prompt)
                    if answer:
                        return {
                            "answer": answer,
                            "source": "gemini",
                            "model": model,
                            "error": None,
                        }
                except Exception as exc:
                    last_error = (
                        f"{model} @ Gemini API: "
                        f"{exc.__class__.__name__}: {exc}"
                    )

        if self.groq_api_key:
            for model in self.groq_models:
                try:
                    answer = self._call_groq(model=model, prompt=prompt)
                    if answer:
                        return {
                            "answer": answer,
                            "source": "groq",
                            "model": model,
                            "error": None,
                        }
                except Exception as exc:
                    last_error = (
                        f"{model} @ {self.groq_url}: "
                        f"{exc.__class__.__name__}: {exc}"
                    )

        for model in self.model_candidates:
            try:
                answer = self._call_ollama(model=model, prompt=prompt)
                if answer:
                    return {
                        "answer": answer,
                        "source": "ollama",
                        "model": model,
                        "error": None,
                    }
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = (
                    f"{model} @ {self.ollama_url}: "
                    f"{exc.__class__.__name__}: {exc}"
                )

        return {
            "answer": fallback_answer,
            "source": "fallback",
            "model": "RAG template",
            "error": last_error,
        }

    def _call_gemini(self, model, prompt):
        url = self.gemini_url_template.format(model=quote(model, safe=""))
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}key={self.gemini_api_key}"
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are a Korean furniture recommendation assistant. "
                            "Use the provided RAG context and answer in practical Korean."
                        )
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.35, "maxOutputTokens": 700},
        }
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=self.gemini_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        if data.get("error"):
            raise ValueError(data["error"])

        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError(f"{model} returned no candidates.")

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        answer = "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()
        if not answer:
            raise ValueError(f"{model} returned an empty response.")
        return answer

    def _call_groq(self, model, prompt):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Korean furniture recommendation assistant. "
                        "Use the provided RAG context and answer in practical Korean."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.35,
            "max_tokens": 700,
        }
        request = Request(
            self.groq_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key}",
            },
            method="POST",
        )

        with urlopen(request, timeout=self.groq_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        if data.get("error"):
            raise ValueError(data["error"])

        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"{model} returned no choices.")

        message = choices[0].get("message") or {}
        answer = (message.get("content") or "").strip()
        if not answer:
            raise ValueError(f"{model} returned an empty response.")
        return answer

    def _call_ollama(self, model, prompt):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.35,
                "top_p": 0.9,
                "num_predict": self.ollama_num_predict,
                "num_ctx": self.ollama_num_ctx,
            },
        }
        request = Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=self.ollama_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        if data.get("error"):
            raise ValueError(data["error"])

        answer = data.get("response", "").strip()
        if not answer:
            raise ValueError(f"{model} returned an empty response.")
        return answer

    def _build_ollama_prompt(
        self,
        question,
        retrieved_docs,
        category_ko=None,
        category_en=None,
        confidence=None,
    ):
        context = "\n\n".join(
            f"[{doc.title}]\n{doc.content}" for doc in retrieved_docs
        )
        if not context:
            context = "검색된 RAG 문서가 없습니다."

        category_text = "분류 결과 없음"
        if category_ko and category_en:
            category_text = f"{category_ko}({category_en})"

        confidence_text = "확률 정보 없음"
        if confidence is not None:
            confidence_text = f"{confidence * 100:.1f}%"

        return f"""
너는 가구 쇼핑몰의 이미지 기반 추천 상담 AI다.
반드시 아래 RAG 문서와 이미지 분류 결과를 근거로 답한다.
근거가 부족하면 확정적으로 말하지 말고, 참고용 추천이라고 표현한다.
Answer in Korean, within 3 short practical sentences.
불필요한 이모티콘이나 장식 문자는 사용하지 않는다.

[이미지 분류 결과]
- 예측 카테고리: {category_text}
- 예측 확률: {confidence_text}

[사용자 질문]
{question}

[RAG 검색 문서]
{context}

[답변 형식]
- Sentence 1: identify the furniture image briefly.
- Sentence 2: recommend placement or style.
- Sentence 3: mention one caution or extra check.
""".strip()

    def _template_answer(
        self,
        question,
        retrieved_docs,
        category_ko=None,
        category_en=None,
        confidence=None,
    ):
        category_text = "현재 이미지"
        if category_ko and category_en:
            category_text = f"{category_ko}({category_en})"

        confidence_note = ""
        if confidence is not None and confidence < 0.6:
            confidence_note = (
                "예측 확률이 아주 높지는 않으므로, 현재 카테고리는 참고용으로 보는 것이 좋습니다.\n\n"
            )

        evidence = "\n".join(f"- {doc.content}" for doc in retrieved_docs)
        if not evidence:
            evidence = "- 관련 문서가 충분히 검색되지 않았습니다."

        return f"""
현재 분석 결과를 기준으로 보면, {category_text} 계열의 가구로 판단됩니다.

{confidence_note}질문: {question}

검색된 RAG 문서를 참고하면 다음 내용을 근거로 추천할 수 있습니다.

{evidence}

정리하면, 공간의 크기와 사용 목적을 먼저 정하고 소재, 색상, 배치 조합을 함께 고려하는 방식이 좋습니다. 현재 답변은 로컬 Ollama 연결이 없을 때 제공되는 RAG 기반 대체 답변입니다.
""".strip()

