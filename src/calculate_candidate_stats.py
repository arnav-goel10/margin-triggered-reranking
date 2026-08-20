#!/usr/bin/env python3
"""
Calculate statistics about score differences between candidate 1 and candidate 2 (and candidate 3):
1. Difference stats (mean & percentiles) for candidate1 - candidate2
2. Same metrics when gold matches candidate1 / candidate2
3. Difference stats for candidate1 - candidate3
"""
import argparse
import json
import re
import string
from pathlib import Path
from collections import defaultdict

import numpy as np


def normalize_answer(s):
    # Lower text and remove punctuation, articles and extra whitespace
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def load_gold_answers(dev_path: str):
    # Load qid -> list of normalized gold answer texts
    with open(dev_path, "r", encoding="utf-8") as f:
        squad_dict = json.load(f)
    
    qid_to_gold = {}
    for article in squad_dict["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                gold_texts = [a["text"] for a in qa.get("answers", [])]
                # Normalize all gold answers
                normalized_golds = [normalize_answer(g) for g in gold_texts if g]
                if not normalized_golds:
                    normalized_golds = [""]  # No-answer questions
                qid_to_gold[qid] = normalized_golds
    return qid_to_gold


def load_candidates(nbest_path: str):
    # Load qid -> list of candidates with text and score
    with open(nbest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate score difference statistics between candidate 1 and 2"
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        required=True,
        help="Path to dev-v1.1.json"
    )
    parser.add_argument(
        "--nbest_file",
        type=str,
        required=True,
        help="Path to predictions_with_5.json (qid -> list of candidates)"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Optional: write results to JSON file"
    )
    args = parser.parse_args()

    print("Loading gold answers...")
    qid_to_gold = load_gold_answers(args.dev_file)
    print(f"Loaded {len(qid_to_gold)} questions")

    print("Loading candidates...")
    qid_to_candidates = load_candidates(args.nbest_file)
    print(f"Loaded candidates for {len(qid_to_candidates)} qids")

    # Statistics
    all_diffs_12 = []
    gold_is_c1_diffs_12 = []
    gold_is_c2_diffs_12 = []

    all_diffs_13 = []
    gold_is_c3_diffs_13 = []

    missing_qids = 0
    no_c2 = 0
    no_c3 = 0

    for qid in qid_to_candidates:
        if qid not in qid_to_gold:
            missing_qids += 1
            continue

        candidates = qid_to_candidates[qid]
        if len(candidates) < 2:
            no_c2 += 1
            continue

        c1 = candidates[0]
        c2 = candidates[1]
        c1_score = float(c1.get("score", 0.0))
        c2_score = float(c2.get("score", 0.0))
        diff12 = c1_score - c2_score

        all_diffs_12.append(diff12)

        # Normalize candidate texts for matching
        c1_text_norm = normalize_answer(c1.get("text", ""))
        c2_text_norm = normalize_answer(c2.get("text", ""))
        gold_texts_norm = qid_to_gold[qid]

        # Check if gold matches candidate 1
        gold_matches_c1 = any(normalize_answer(g) == c1_text_norm for g in gold_texts_norm)
        # Check if gold matches candidate 2
        gold_matches_c2 = any(normalize_answer(g) == c2_text_norm for g in gold_texts_norm)

        if gold_matches_c1:
            gold_is_c1_diffs_12.append(diff12)
        elif gold_matches_c2:
            gold_is_c2_diffs_12.append(diff12)

        # Candidate 3 stats if available
        if len(candidates) >= 3:
            c3 = candidates[2]
            c3_score = float(c3.get("score", 0.0))
            diff13 = c1_score - c3_score
            all_diffs_13.append(diff13)

            c3_text_norm = normalize_answer(c3.get("text", ""))
            gold_matches_c3 = any(normalize_answer(g) == c3_text_norm for g in gold_texts_norm)

            if gold_matches_c3:
                gold_is_c3_diffs_13.append(diff13)
        else:
            no_c3 += 1

    # helper to summarize stats
    def summarize(values):
        if not values:
            return {
                "count": 0,
                "avg": 0.0,
                "p25": None,
                "median": None,
                "p75": None,
            }
        arr = np.array(values, dtype=np.float64)
        return {
            "count": len(arr),
            "avg": float(arr.mean()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }

    overall_stats_12 = summarize(all_diffs_12)
    gold_c1_stats_12 = summarize(gold_is_c1_diffs_12)
    gold_c2_stats_12 = summarize(gold_is_c2_diffs_12)

    overall_stats_13 = summarize(all_diffs_13)
    gold_c3_stats_13 = summarize(gold_is_c3_diffs_13)

    results = {
        "candidate1_minus_candidate2": {
            "overall": overall_stats_12,
            "gold_is_candidate1": gold_c1_stats_12,
            "gold_is_candidate2": gold_c2_stats_12,
        },
        "candidate1_minus_candidate3": {
            "overall": overall_stats_13,
            "gold_is_candidate3": gold_c3_stats_13,
        },
        "counts": {
            "total_questions_with_c1_and_c2": len(all_diffs_12),
            "questions_where_gold_is_c1": len(gold_is_c1_diffs_12),
            "questions_where_gold_is_c2": len(gold_is_c2_diffs_12),
            "total_questions_with_c1_and_c3": len(all_diffs_13),
            "questions_where_gold_is_c3": len(gold_is_c3_diffs_13),
            "missing_qids": missing_qids,
            "questions_without_c2": no_c2,
            "questions_without_c3": no_c3,
        }
    }

    print("\n" + "="*60)
    print("SCORE DIFFERENCE STATISTICS")
    print("="*60)
    
    def print_stats(label, stats):
        print(f"{label}:")
        print(f"  • count   : {stats['count']}")
        print(f"  • avg     : {stats['avg']:.6f}" if stats['count'] else "  • avg     : n/a")
        print(f"  • p25     : {stats['p25']:.6f}" if stats['p25'] is not None else "  • p25     : n/a")
        print(f"  • median  : {stats['median']:.6f}" if stats['median'] is not None else "  • median  : n/a")
        print(f"  • p75     : {stats['p75']:.6f}" if stats['p75'] is not None else "  • p75     : n/a")

    print("Candidate1 − Candidate2")
    print_stats("  1. Overall", overall_stats_12)
    print_stats("  2. Gold is candidate 1", gold_c1_stats_12)
    print_stats("  3. Gold is candidate 2", gold_c2_stats_12)

    print("\nCandidate1 − Candidate3")
    print_stats("  4. Overall", overall_stats_13)
    print_stats("  5. Gold is candidate 3", gold_c3_stats_13)

    print("\nCounts:")
    print(f"  • Questions with C1 & C2: {len(all_diffs_12)}")
    print(f"  • Questions where gold matches C1: {len(gold_is_c1_diffs_12)}")
    print(f"  • Questions where gold matches C2: {len(gold_is_c2_diffs_12)}")
    print(f"  • Questions with C1 & C3: {len(all_diffs_13)}")
    print(f"  • Questions where gold matches C3: {len(gold_is_c3_diffs_13)}")
    print(f"  • Missing QIDs in gold: {missing_qids}")
    print(f"  • Questions without C2: {no_c2}")
    print(f"  • Questions without C3: {no_c3}")
    print("="*60)

    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.out_file}")


if __name__ == "__main__":
    main()

