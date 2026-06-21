"""Task 3: RAG core logic — retriever + prompt + generator."""

from __future__ import annotations

from typing import Optional

from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
import chromadb

from src.embedder import load_chroma_store, load_embedding_model, query_store


PROMPT_TEMPLATE = """\
You are a financial analyst assistant for CrediTrust Financial. Your task is to \
answer questions about customer complaints submitted to the CFPB. Use ONLY the \
retrieved complaint excerpts below to formulate your answer. Cite specific issues \
mentioned by customers where possible. If the provided context does not contain \
enough information to answer the question, say so explicitly — do not speculate.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(
        self,
        chroma_dir: str = "vector_store/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        k: int = 3,
        max_new_tokens: int = 150,
    ):
        self.k = k
        self.collection: chromadb.Collection = load_chroma_store(chroma_dir)
        self.embedder: SentenceTransformer = load_embedding_model(embedding_model)
        self._llm_model_name = llm_model
        self._max_new_tokens = max_new_tokens
        self._generator = None

    def _get_generator(self):
        if self._generator is None:
            self._generator = hf_pipeline(
                "text-generation",
                model=self._llm_model_name,
                device_map="cpu",
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        return self._generator

    def retrieve(
        self, question: str, product_filter: Optional[str] = None
    ) -> list[dict]:
        return query_store(
            self.collection,
            question,
            self.embedder,
            k=self.k,
            product_filter=product_filter,
        )

    def build_prompt(self, question: str, hits: list[dict], max_chunk_chars: int = 300) -> str:
        context_parts = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            product = meta.get("product_category", "Unknown")
            issue = meta.get("issue", "")
            header = f"[{i}] Product: {product}" + (f" | Issue: {issue}" if issue else "")
            text = hit["text"][:max_chunk_chars]
            context_parts.append(f"{header}\n{text}")
        context = "\n\n".join(context_parts)
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, prompt: str) -> str:
        gen = self._get_generator()
        output = gen(prompt)[0]["generated_text"]
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip()
        return output[len(prompt):].strip()

    def run(
        self, question: str, product_filter: Optional[str] = None
    ) -> dict:
        """End-to-end RAG: retrieve → prompt → generate."""
        hits = self.retrieve(question, product_filter)
        prompt = self.build_prompt(question, hits)
        answer = self.generate(prompt)
        return {"answer": answer, "sources": hits}
