from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.config import get_settings
from app.core.vectorstore import RetrievalFilters
from app.services import Services


@dataclass(frozen=True)
class QuestionSpec:
    question: str
    expected_answer_contains: str
    expected_doc_id: str
    expected_chunk_keywords: list[str]


@dataclass(frozen=True)
class QuestionResult:
    question: str
    precision: float
    recall_at_k: float
    generation_latency_ms: int
    latency_ms: int
    faithfulness: float
    context_utilisation: float
    answer: str


def _load_questions(path: Path) -> list[QuestionSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [QuestionSpec(**item) for item in raw]


def _tokenize(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
    tokens = {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords}
    return tokens


def _score_retrieval(question: QuestionSpec, chunks: Sequence[object]) -> tuple[float, float]:
    keywords = [
        keyword.lower()
        for keyword in question.expected_chunk_keywords
        if keyword and keyword != "TODO"
    ]
    if question.expected_doc_id == "TODO" and not keywords:
        return 0.0, 0.0
    relevant_count = 0
    for chunk in chunks:
        text = getattr(chunk, "text", "").lower()
        doc_id = getattr(chunk, "doc_id", "")
        if question.expected_doc_id != "TODO" and doc_id != question.expected_doc_id:
            continue
        if keywords and not all(keyword in text for keyword in keywords):
            continue
        relevant_count += 1
    precision = relevant_count / max(len(chunks), 1)
    recall_at_k = 1.0 if relevant_count > 0 else 0.0
    return precision, recall_at_k


def _score_answer(answer: str, context_text: str) -> tuple[float, float]:
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(context_text)
    if not answer_tokens or not context_tokens:
        return 0.0, 0.0
    overlap = answer_tokens.intersection(context_tokens)
    faithfulness = len(overlap) / len(answer_tokens)
    context_utilisation = len(overlap) / len(context_tokens)
    return faithfulness, context_utilisation


def run_evaluation(questions_path: Path) -> dict[str, object]:
    settings = get_settings()
    services = Services.build(settings)
    questions = _load_questions(questions_path)

    results: list[QuestionResult] = []
    for item in questions:
        filters = (
            None
            if item.expected_doc_id == "TODO"
            else RetrievalFilters(doc_id=item.expected_doc_id)
        )
        retrieval = services.retriever.retrieve(
            item.question,
            top_k=settings.retrieval_top_k_default,
            similarity_threshold=settings.retrieval_similarity_threshold,
            filters=filters,
        )
        prompt_bundle = services.prompt_builder.build(item.question, retrieval.chunks)
        generation_start = perf_counter()
        answer = services.generation_client.generate(prompt_bundle.prompt).text
        generation_latency_ms = int((perf_counter() - generation_start) * 1000)
        precision, recall_at_k = _score_retrieval(item, retrieval.chunks)
        context_text = "\n".join(chunk.text for chunk in retrieval.chunks)
        faithfulness, context_utilisation = _score_answer(answer, context_text)
        total_latency_ms = (
            retrieval.embedding_latency_ms + retrieval.retrieval_latency_ms + generation_latency_ms
        )
        results.append(
            QuestionResult(
                question=item.question,
                precision=precision,
                recall_at_k=recall_at_k,
                generation_latency_ms=generation_latency_ms,
                latency_ms=total_latency_ms,
                faithfulness=faithfulness,
                context_utilisation=context_utilisation,
                answer=answer,
            )
        )

    summary = {
        "questions": len(results),
        "retrieval_precision": mean(result.precision for result in results) if results else 0.0,
        "recall_at_k": mean(result.recall_at_k for result in results) if results else 0.0,
        "generation_latency_ms": mean(result.generation_latency_ms for result in results)
        if results
        else 0.0,
        "latency_ms": mean(result.latency_ms for result in results) if results else 0.0,
        "answer_faithfulness": mean(result.faithfulness for result in results) if results else 0.0,
        "context_utilisation": mean(result.context_utilisation for result in results)
        if results
        else 0.0,
    }
    return {
        "summary": summary,
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PDF RAG evaluation suite.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_evaluation(args.questions)
    output = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
