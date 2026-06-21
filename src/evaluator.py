"""Task 3: Qualitative evaluation utilities for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.rag_pipeline import RAGPipeline


EVAL_QUESTIONS = [
    "Why are customers unhappy with their credit cards?",
    "What are the most common issues with personal loans?",
    "What problems do customers report about savings accounts?",
    "What complaints exist about money transfers?",
    "Are there complaints about unauthorized transactions on credit cards?",
    "What do customers say about loan repayment or billing issues?",
    "Are there fraud-related complaints across any product?",
    "What issues do customers raise about customer service?",
    "Are there complaints about fees or charges being unexpected?",
    "What are the most critical unresolved complaints?",
]


@dataclass
class EvalResult:
    question: str
    answer: str
    sources: list[dict]
    score: Optional[int] = None
    comments: str = ""

    def top_sources(self, n: int = 2) -> list[str]:
        return [s["text"][:150] + "…" for s in self.sources[:n]]


def run_evaluation(pipeline: RAGPipeline, questions: list[str] | None = None) -> list[EvalResult]:
    questions = questions or EVAL_QUESTIONS
    results = []
    for q in questions:
        print(f"Evaluating: {q}")
        out = pipeline.run(q)
        results.append(EvalResult(question=q, answer=out["answer"], sources=out["sources"]))
    return results


def results_to_markdown(results: list[EvalResult]) -> str:
    header = (
        "| # | Question | Generated Answer | Retrieved Sources | Score | Comments |\n"
        "|---|----------|-----------------|-------------------|-------|----------|\n"
    )
    rows = []
    for i, r in enumerate(results, 1):
        sources_md = "<br>".join(r.top_sources())
        score = r.score if r.score is not None else "—"
        rows.append(
            f"| {i} | {r.question} | {r.answer[:200]}… | {sources_md} | {score} | {r.comments} |"
        )
    return header + "\n".join(rows)
