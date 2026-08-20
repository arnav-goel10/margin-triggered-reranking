#!/usr/bin/env python3
import argparse
import collections
import json
import re
import string
from pathlib import Path


def normalize_answer(s: str) -> str:
    # normalize answer text
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def load_gold_answers(dev_path: str) -> dict[str, list[str]]:
    with open(dev_path, "r") as f:
        data = json.load(f)
    qid_to_answers = {}
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                gold_answers = [a["text"] for a in qa["answers"] if normalize_answer(a["text"])]
                if not gold_answers:
                    gold_answers = [""]
                qid_to_answers[qid] = gold_answers
    return qid_to_answers


def determine_k(predictions: dict) -> int:
    # figure out k from first list we find
    for v in predictions.values():
        if isinstance(v, list):
            return len(v)
    raise ValueError("Predictions file must map qid -> list of candidate dicts")


def compute_position_stats(
    gold: dict[str, list[str]],
    preds: dict,
):
    total = len(gold)
    k = determine_k(preds)

    # Per-position EM counts
    pos_em_counts = [0] * k
    # Exclusive per-position EM (first correct at position i)
    exclusive_em_counts = [0] * k

    # Cumulative EM: any of top i is exact
    cum_em_counts = [0] * k

    # Cumulative best-F1: avg over questions of max F1 among top i
    cum_best_f1_sums = [0.0] * k

    # Cumulative any-F1>0 counts
    cum_any_f1_pos_counts = [0] * k

    # Score gap tracking when gold is within top-3
    diffs_1_2_when_gold_in_top3: list[float] = []
    diffs_2_3_when_gold_in_top3: list[float] = []

    for qid, gold_answers in gold.items():
        if qid not in preds:
            continue
        candidates = preds[qid]
        # candidates expected as list of dicts with "text" field
        cand_texts = []
        cand_scores = []
        for c in candidates:
            if isinstance(c, dict):
                cand_texts.append(str(c.get("text", "")))
                score_val = c.get("score", None)
                try:
                    cand_scores.append(float(score_val) if score_val is not None else None)
                except (TypeError, ValueError):
                    cand_scores.append(None)
            else:
                cand_texts.append(str(c))
                cand_scores.append(None)

        # Per position EM
        for i in range(min(k, len(cand_texts))):
            em_i = max(compute_exact(ga, cand_texts[i]) for ga in gold_answers)
            pos_em_counts[i] += em_i

        # Cumulative stats
        seen_exact = False
        best_f1_so_far = 0.0
        any_f1_pos = False
        first_exact_rank = None
        for i in range(min(k, len(cand_texts))):
            em_i = max(compute_exact(ga, cand_texts[i]) for ga in gold_answers)
            if em_i:
                seen_exact = True
                if first_exact_rank is None:
                    first_exact_rank = i

            best_f1_i = max(compute_f1(ga, cand_texts[i]) for ga in gold_answers)
            best_f1_so_far = max(best_f1_so_far, best_f1_i)
            if best_f1_i > 0.0:
                any_f1_pos = True

            if seen_exact:
                cum_em_counts[i] += 1
            cum_best_f1_sums[i] += best_f1_so_far
            if any_f1_pos:
                cum_any_f1_pos_counts[i] += 1

        if first_exact_rank is not None:
            exclusive_em_counts[first_exact_rank] += 1

        # Score diffs when gold is in top-3
        top_n = min(3, len(cand_texts))
        if top_n >= 1:
            gold_in_top3 = any(max(compute_exact(ga, cand_texts[i]) for ga in gold_answers) for i in range(top_n))
            if gold_in_top3:
                if top_n >= 2 and cand_scores[0] is not None and cand_scores[1] is not None:
                    diffs_1_2_when_gold_in_top3.append(cand_scores[0] - cand_scores[1])
                if top_n >= 3 and cand_scores[1] is not None and cand_scores[2] is not None:
                    diffs_2_3_when_gold_in_top3.append(cand_scores[1] - cand_scores[2])

    # convert to percentages
    pos_em_pct = [100.0 * c / total for c in pos_em_counts]
    cum_em_pct = [100.0 * c / total for c in cum_em_counts]
    cum_avg_best_f1_pct = [100.0 * (s / total) for s in cum_best_f1_sums]
    cum_any_f1_pos_pct = [100.0 * c / total for c in cum_any_f1_pos_counts]

    exclusive_em_pct = [100.0 * c / total for c in exclusive_em_counts]

    return {
        "total": total,
        "k": k,
        "position_em_percent": pos_em_pct,                # len k
        "exclusive_position_em_percent": exclusive_em_pct, # len k
        "cumulative_em_percent": cum_em_pct,              # len k
        "cumulative_avg_best_f1_percent": cum_avg_best_f1_pct,  # len k
        "cumulative_any_f1_positive_percent": cum_any_f1_pos_pct,  # len k
        "avg_score_diff_1_2_when_gold_in_top3": (sum(diffs_1_2_when_gold_in_top3) / len(diffs_1_2_when_gold_in_top3)) if diffs_1_2_when_gold_in_top3 else None,
        "avg_score_diff_2_3_when_gold_in_top3": (sum(diffs_2_3_when_gold_in_top3) / len(diffs_2_3_when_gold_in_top3)) if diffs_2_3_when_gold_in_top3 else None,
        "count_gold_in_top3_for_1_2": len(diffs_1_2_when_gold_in_top3),
        "count_gold_in_top3_for_2_3": len(diffs_2_3_when_gold_in_top3),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    default_dev = str(root / "data" / "dev-v1.1.json")
    default_pred = str(root / "predictions" / "predictions_with_5.json")

    parser = argparse.ArgumentParser(
        description="Compute per-position and cumulative EM/F1 stats for top-k predictions"
    )
    parser.add_argument("--dev_file", type=str, default=default_dev, help="Path to SQuAD dev-v1.1.json")
    parser.add_argument("--predictions_file", type=str, default=default_pred, help="Path to predictions_with_k.json (qid -> list of candidates)")
    parser.add_argument("--out_file", type=str, default="", help="Optional path to write JSON metrics")
    args = parser.parse_args()

    gold = load_gold_answers(args.dev_file)
    with open(args.predictions_file, "r") as f:
        preds = json.load(f)

    metrics = compute_position_stats(gold, preds)

    # Print results
    total = metrics["total"]
    k = metrics["k"]
    print(f"Questions: {total}, k={k}")
    print("Per-position EM (%):", [round(x, 2) for x in metrics["position_em_percent"]])
    print("Exclusive per-position EM (%):", [round(x, 2) for x in metrics["exclusive_position_em_percent"]])
    print("Cumulative EM (%):  ", [round(x, 2) for x in metrics["cumulative_em_percent"]])
    print(
        "Cumulative avg best-F1 (%):",
        [round(x, 2) for x in metrics["cumulative_avg_best_f1_percent"]],
    )
    print(
        "Cumulative any F1>0 (%):   ",
        [round(x, 2) for x in metrics["cumulative_any_f1_positive_percent"]],
    )

    if args.out_file:
        with open(args.out_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {args.out_file}")


if __name__ == "__main__":
    main()


