#!/usr/bin/env python3
"""Count how often the gold answer appears among the top-k candidates."""
import argparse
import json
import re
import string
from pathlib import Path


def normalize_answer(s: str) -> str:
    # Lower text and remove punctuation, articles and extra whitespace
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def load_gold_answers(dev_path: str):
    with open(dev_path, "r", encoding="utf-8") as f:
        js = json.load(f)

    qid_to_gold = {}
    for article in js["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                gold_texts = [a["text"] for a in qa.get("answers", [])]
                normalized = [normalize_answer(g) for g in gold_texts if g]
                if not normalized:
                    normalized = [""]
                qid_to_gold[qid] = normalized
    return qid_to_gold


def load_candidates(nbest_path: str):
    with open(nbest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Count gold answer positions within top-k candidates")
    parser.add_argument("--dev_file", type=str, required=True, help="Path to dev-v1.1.json")
    parser.add_argument("--nbest_file", type=str, required=True, help="Path to predictions_with_5.json")
    parser.add_argument("--max_k", type=int, default=5, help="Number of top candidates to inspect")
    parser.add_argument("--out_file", type=str, default=None, help="Optional path to write JSON summary")
    args = parser.parse_args()

    print("Loading gold answers...")
    qid_to_gold = load_gold_answers(args.dev_file)
    print(f"Loaded {len(qid_to_gold)} questions")

    print("Loading candidate lists...")
    qid_to_candidates = load_candidates(args.nbest_file)
    print(f"Loaded candidates for {len(qid_to_candidates)} questions")

    max_k = max(1, args.max_k)
    counts = {str(i): 0 for i in range(1, max_k + 1)}
    counts["not_in_top_k"] = 0
    counts["missing_candidates"] = 0

    total_questions = 0

    for qid, gold_norms in qid_to_gold.items():
        total_questions += 1
        candidates = qid_to_candidates.get(qid)
        if not candidates:
            counts["missing_candidates"] += 1
            counts["not_in_top_k"] += 1
            continue

        found_position = None
        for idx, cand in enumerate(candidates[:max_k], start=1):
            cand_text_norm = normalize_answer(cand.get("text", ""))
            if any(cand_text_norm == g for g in gold_norms):
                found_position = idx
                break

        if found_position is None:
            counts["not_in_top_k"] += 1
        else:
            counts[str(found_position)] += 1

    print("\n" + "=" * 60)
    print(f"Gold Answer Positions (top-{max_k})")
    print("=" * 60)
    for i in range(1, max_k + 1):
        print(f"Position {i}: {counts[str(i)]}")
    print(f"Not in top-{max_k}: {counts['not_in_top_k']}")
    print(f"Questions missing candidate list: {counts['missing_candidates']}")
    print(f"Total questions: {total_questions}")
    print("=" * 60)

    if args.out_file:
        output = {
            "max_k": max_k,
            "counts": counts,
            "total_questions": total_questions,
        }
        with open(args.out_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.out_file}")


if __name__ == "__main__":
    main()
