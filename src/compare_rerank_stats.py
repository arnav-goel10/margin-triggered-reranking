#!/usr/bin/env python3
# Compare reranked predictions against original top-1 candidates
# Outputs counts of how many questions changed, and among those changes how often
# we moved from incorrect→correct, correct→incorrect, etc.
import argparse
import json
import re
import string
from typing import Dict, List


def normalize_answer(s: str) -> str:
    # Lower text and remove punctuation, articles and extra whitespace
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def load_gold_answers(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qid_to_golds: Dict[str, List[str]] = {}
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                answers = [normalize_answer(a["text"]) for a in qa.get("answers", []) if a.get("text")]
                if not answers:
                    answers = [""]
                qid_to_golds[qid] = answers
    return qid_to_golds


def load_nbest(path: str) -> Dict[str, List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    return {str(k): (v if isinstance(v, str) else str(v)) for k, v in preds.items()}


def is_correct(answer: str, golds: List[str]) -> bool:
    norm = normalize_answer(answer)
    return any(norm == g for g in golds)


def main():
    ap = argparse.ArgumentParser(description="Compare reranked predictions vs original top1 candidates")
    ap.add_argument("--dev_file", required=True, help="Path to SQuAD dev file for gold answers")
    ap.add_argument("--nbest_file", required=True, help="Top-k predictions JSON (original order)")
    ap.add_argument("--reranked_file", required=True, help="Reranked top-1 predictions JSON")
    ap.add_argument("--out_file", default=None, help="Optional JSON file to write stats")
    args = ap.parse_args()

    print("Loading gold answers...")
    qid_to_gold = load_gold_answers(args.dev_file)
    print(f"Loaded {len(qid_to_gold)} gold entries")

    print("Loading top-k candidates...")
    qid_to_candidates = load_nbest(args.nbest_file)
    print(f"Loaded {len(qid_to_candidates)} nbest entries")

    print("Loading reranked predictions...")
    reranked = load_predictions(args.reranked_file)
    print(f"Loaded {len(reranked)} reranked entries")

    counts = {
        "total": 0,
        "missing_in_nbest": 0,
        "missing_in_gold": 0,
        "unchanged": 0,
        "changed": 0,
        "changed_orig_incorrect_new_correct": 0,
        "changed_orig_correct_new_incorrect": 0,
        "changed_orig_incorrect_new_incorrect": 0,
        "changed_orig_correct_new_correct": 0,
    }

    for qid, new_answer in reranked.items():
        if qid not in qid_to_gold:
            counts["missing_in_gold"] += 1
            continue
        if qid not in qid_to_candidates:
            counts["missing_in_nbest"] += 1
            continue

        candidates = qid_to_candidates[qid]
        if not candidates:
            counts["missing_in_nbest"] += 1
            continue

        original_answer = candidates[0].get("text", "")
        golds = qid_to_gold[qid]

        orig_correct = is_correct(original_answer, golds)
        new_correct = is_correct(new_answer, golds)

        counts["total"] += 1

        if normalize_answer(original_answer) == normalize_answer(new_answer):
            counts["unchanged"] += 1
            continue

        counts["changed"] += 1

        if not orig_correct and new_correct:
            counts["changed_orig_incorrect_new_correct"] += 1
        elif orig_correct and not new_correct:
            counts["changed_orig_correct_new_incorrect"] += 1
        elif not orig_correct and not new_correct:
            counts["changed_orig_incorrect_new_incorrect"] += 1
        else:  # orig_correct and new_correct
            counts["changed_orig_correct_new_correct"] += 1

    print("\n" + "=" * 60)
    print("RERANK VS ORIGINAL TOP-1 STATS")
    print("=" * 60)
    print(f"Total comparable questions: {counts['total']}")
    print(f"Unchanged top-1 answer:     {counts['unchanged']}")
    print(f"Changed top-1 answer:       {counts['changed']}")
    print()
    if counts["changed"]:
        print("Changed breakdown:")
        print(f"  • orig incorrect → new correct : {counts['changed_orig_incorrect_new_correct']}")
        print(f"  • orig correct → new incorrect : {counts['changed_orig_correct_new_incorrect']}")
        print(f"  • both incorrect               : {counts['changed_orig_incorrect_new_incorrect']}")
        print(f"  • both correct                 : {counts['changed_orig_correct_new_correct']}")
    else:
        print("No questions changed their top-1 answer.")
    print("=" * 60)
    if counts["missing_in_gold"] or counts["missing_in_nbest"]:
        print(f"Skipped due to missing gold entries: {counts['missing_in_gold']}")
        print(f"Skipped due to missing nbest entries: {counts['missing_in_nbest']}")
        print("=" * 60)

    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=2)
        print(f"Stats written to {args.out_file}")


if __name__ == "__main__":
    main()
