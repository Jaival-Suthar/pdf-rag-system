from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median


INPUT = Path("eval/results/m1-rerank-k20-v3.json")
OUTPUT = Path("eval/results/rq3-score-margin-analysis-v1.json")


def as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def as_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    return data

def classify_structural_chunk(chunk: dict) -> str:
    text = str(chunk.get("text", "")).strip().lower()
    section_title = str(chunk.get("section_title", "")).strip().lower()

    haystack = f"{section_title} {text}"

    if (
        "table of contents" in haystack
        or section_title == "contents"
        or text.startswith("contents")
    ):
        return "CONTENTS"

    if (
        section_title == "index"
        or text.startswith("index")
        or "subject index" in haystack
    ):
        return "INDEX"

    if "introduction" in section_title or text.startswith("introduction"):
        return "INTRODUCTION"

    if "copyright" in haystack:
        return "COPYRIGHT"

    if "glossary" in section_title or text.startswith("glossary"):
        return "GLOSSARY"

    if (
        "bibliography" in section_title
        or text.startswith("bibliography")
        or "references" in section_title
    ):
        return "BIBLIOGRAPHY"

    if any(
        marker in haystack
        for marker in (
            "preface",
            "foreword",
            "acknowledgments",
            "acknowledgements",
            "about the author",
        )
    ):
        return "STRUCTURAL_OTHER"

    return "SUBSTANTIVE"

def main() -> None:
    report = load_json(INPUT)

    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("Expected benchmark JSON to contain a 'results' list")

    failures: list[dict[str, object]] = []

    for result in results:
        if not isinstance(result, dict):
            continue

        question_id = str(result.get("question_id", ""))

        candidate_chunks = result.get("candidate_chunks", [])
        final_chunks = result.get("retrieved_chunks", [])

        if not isinstance(candidate_chunks, list):
            continue

        if not isinstance(final_chunks, list):
            continue

        candidate_by_id: dict[str, dict] = {}
        final_by_id: dict[str, dict] = {}

        for chunk in candidate_chunks:
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id")
                if isinstance(chunk_id, str):
                    candidate_by_id[chunk_id] = chunk

        for rank, chunk in enumerate(final_chunks, start=1):
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id")
                if isinstance(chunk_id, str):
                    final_by_id[chunk_id] = chunk

        gold_evidence = result.get("gold_evidence", [])

        if not isinstance(gold_evidence, list):
            continue

        gold_ids = {
            evidence.get("chunk_id")
            for evidence in gold_evidence
            if isinstance(evidence, dict)
            and isinstance(evidence.get("chunk_id"), str)
        }

        for gold_id in gold_ids:
            gold = candidate_by_id.get(gold_id)

            # Gold evidence never entered candidate-20.
            if gold is None:
                continue

            # Gold survived into final top-k.
            if gold_id in final_by_id:
                continue

            gold_candidate_rank = as_int(gold.get("candidate_rank"))
            gold_dense_score = as_float(gold.get("raw_score"))
            gold_reranker_score = as_float(gold.get("reranker_score"))

            # Find the final chunks that displaced the gold chunk.
            replacements: list[dict[str, object]] = []

            for final_rank, replacement in enumerate(final_chunks, start=1):
                if not isinstance(replacement, dict):
                    continue

                replacement_id = replacement.get("chunk_id")

                if not isinstance(replacement_id, str):
                    continue

                if replacement_id == gold_id:
                    continue

                replacement_candidate = candidate_by_id.get(replacement_id)

                if replacement_candidate is None:
                    continue

                replacement_reranker_score = as_float(
                    replacement.get("reranker_score")
                )

                if replacement_reranker_score is None:
                    replacement_reranker_score = as_float(
                        replacement_candidate.get("reranker_score")
                    )

                replacement_dense_score = as_float(
                    replacement_candidate.get("raw_score")
                )

                replacement_candidate_rank = as_int(
                    replacement_candidate.get("candidate_rank")
                )

                score_margin = None
                if (
                    gold_reranker_score is not None
                    and replacement_reranker_score is not None
                ):
                    score_margin = (
                        replacement_reranker_score - gold_reranker_score
                    )

                dense_rank_delta = None
                if (
                    gold_candidate_rank is not None
                    and replacement_candidate_rank is not None
                ):
                    dense_rank_delta = (
                        replacement_candidate_rank - gold_candidate_rank
                    )

                replacement_category = classify_structural_chunk(
                    replacement_candidate
                )

                replacements.append(
                    {
                        "chunk_id": replacement_id,
                        "final_rank": final_rank,
                        "candidate_rank": replacement_candidate_rank,
                        "dense_score": replacement_dense_score,
                        "reranker_score": replacement_reranker_score,
                        "score_margin": score_margin,
                        "category": replacement_category,
                        "section_title": replacement_candidate.get(
                            "section_title"
                        ),
                        "page_number": replacement_candidate.get(
                            "page_number"
                        ),
                        "text": replacement_candidate.get("text"),
                        "dense_rank_delta_vs_gold": dense_rank_delta,
                    }
                )

            replacements.sort(
                key=lambda item: (
                    item["final_rank"]
                    if isinstance(item.get("final_rank"), int)
                    else 999999
                )
            )

            if not replacements:
                continue

            best_replacement = replacements[0]

            failures.append(
                {
                    "question_id": question_id,
                    "gold_chunk_id": gold_id,
                    "gold_candidate_rank": gold_candidate_rank,
                    "gold_dense_score": gold_dense_score,
                    "gold_reranker_score": gold_reranker_score,
                    "gold_section_title": gold.get("section_title"),
                    "gold_page_number": gold.get("page_number"),
                    "best_replacement": best_replacement,
                    "all_final_replacements": replacements,
                }
            )

    margins = [
        item["best_replacement"]["score_margin"]
        for item in failures
        if isinstance(item.get("best_replacement"), dict)
        and isinstance(item["best_replacement"].get("score_margin"), (int, float))
    ]

    gold_scores = [
        item["gold_reranker_score"]
        for item in failures
        if isinstance(item.get("gold_reranker_score"), (int, float))
    ]

    replacement_scores = [
        item["best_replacement"]["reranker_score"]
        for item in failures
        if isinstance(item.get("best_replacement"), dict)
        and isinstance(item["best_replacement"].get("reranker_score"), (int, float))
    ]

    structural_categories = {
        "CONTENTS",
        "INDEX",
        "INTRODUCTION",
        "COPYRIGHT",
        "GLOSSARY",
        "BIBLIOGRAPHY",
        "STRUCTURAL_OTHER",
    }

    structural_replacements = 0

    for item in failures:
        replacement = item.get("best_replacement")

        if not isinstance(replacement, dict):
            continue

        category = str(replacement.get("category", ""))

        if category in structural_categories:
            structural_replacements += 1

    analysis = {
        "input": str(INPUT),
        "questions_analyzed": len(results),
        "gold_chunks_lost_after_reranking": len(failures),
        "reranker_score_available_for_gold_failures": len(gold_scores),
        "reranker_score_available_for_replacements": len(replacement_scores),
        "score_margin_statistics": {
            "mean": mean(margins) if margins else None,
            "median": median(margins) if margins else None,
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
        "gold_reranker_score_statistics": {
            "mean": mean(gold_scores) if gold_scores else None,
            "median": median(gold_scores) if gold_scores else None,
        },
        "replacement_reranker_score_statistics": {
            "mean": mean(replacement_scores) if replacement_scores else None,
            "median": median(replacement_scores) if replacement_scores else None,
        },
        "structural_best_replacements": structural_replacements,
        "substantive_or_other_best_replacements": (
            len(failures) - structural_replacements
        ),
        "failure_cases": failures,
    }

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2, ensure_ascii=False)

    print(f"Analyzed: {INPUT}")
    print(f"Output:   {OUTPUT}")
    print(f"Gold chunks lost after reranking: {len(failures)}")

    if margins:
        print(f"Mean reranker score margin: {mean(margins):.4f}")
        print(f"Median reranker score margin: {median(margins):.4f}")

    if gold_scores:
        print(f"Mean gold reranker score: {mean(gold_scores):.4f}")

    if replacement_scores:
        print(
            "Mean replacement reranker score: "
            f"{mean(replacement_scores):.4f}"
        )


if __name__ == "__main__":
    main()