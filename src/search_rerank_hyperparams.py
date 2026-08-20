#!/usr/bin/env python3
import argparse
import collections
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from rerank_squad_candidates import (
    dedup_and_filter_candidates,
    make_candidate_text,
    normalize_scores,
)


def parse_label_path(pairs: List[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got: {item}")
        label, path = item.split("=", 1)
        label = label.strip()
        p = Path(path.strip())
        if not label:
            raise ValueError(f"Empty label in: {item}")
        if label in mapping:
            raise ValueError(f"Duplicate label: {label}")
        mapping[label] = p
    return mapping


def load_dev_dataset(dev_path: Path):
    with dev_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    qid_to_question: Dict[str, str] = {}
    qid_to_answers: Dict[str, List[str]] = {}
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                qid_to_question[qid] = qa["question"]
                answers = [ans["text"] for ans in qa.get("answers", []) if ans.get("text")]
                if not answers:
                    answers = [""]
                qid_to_answers[qid] = answers
    return qid_to_question, qid_to_answers


def normalize_answer(text: str) -> str:
    # normalize answer text for comparison
    import re
    import string

    text = text or ""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact_f1(golds: List[str], pred: str) -> Tuple[int, float]:
    if not golds:
        golds = [""]
    max_exact = 0
    max_f1 = 0.0
    for gold in golds:
        exact = int(normalize_answer(gold) == normalize_answer(pred))
        gold_toks = get_tokens(gold)
        pred_toks = get_tokens(pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            f1 = float(int(gold_toks == pred_toks))
        elif num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
        if exact > max_exact:
            max_exact = exact
        if f1 > max_f1:
            max_f1 = f1
    return max_exact, max_f1


def compute_metrics(qid_to_answers: Dict[str, List[str]], preds: Dict[str, str]) -> Tuple[float, float]:
    total = 0
    exact_sum = 0
    f1_sum = 0.0
    for qid, golds in qid_to_answers.items():
        if qid not in preds:
            continue
        total += 1
        exact, f1 = compute_exact_f1(golds, preds[qid])
        exact_sum += exact
        f1_sum += f1
    if total == 0:
        return 0.0, 0.0
    return 100.0 * exact_sum / total, 100.0 * f1_sum / total


def build_entries(
    label: str,
    nbest_path: Path,
    qid_to_question: Dict[str, str],
    qid_to_answers: Dict[str, List[str]],
    normalize_mode: str,
    candidate_text_mode: str,
    max_answer_tokens: int,
    cap_topk: int,
    reranker_type: str,
    batch_size: int,
    bi_model: SentenceTransformer = None,
    cross_model: CrossEncoder = None,
):
    with nbest_path.open("r", encoding="utf-8") as f:
        qid_to_cands = json.load(f)

    entries = []
    skipped_missing_question = 0
    for qid, candidates in qid_to_cands.items():
        if qid not in qid_to_question or qid not in qid_to_answers:
            skipped_missing_question += 1
            continue
        cands = dedup_and_filter_candidates(candidates, max_tokens=max_answer_tokens)
        if cap_topk and len(cands) > cap_topk:
            cands = cands[:cap_topk]
        if not cands:
            entries.append({
                "qid": qid,
                "question": qid_to_question[qid],
                "cands": [],
                "base_scores": np.array([], dtype=np.float32),
                "base_norm": np.array([], dtype=np.float32),
                "aux_norm": np.array([], dtype=np.float32),
                "gap12": math.inf,
                "base_top1": "",
                "base_correct": False,
                "gold_norms": [normalize_answer(g) for g in qid_to_answers[qid]],
            })
            continue

        base_scores = np.array([float(c.get("score", 0.0)) for c in cands], dtype=np.float32)
        base_norm = normalize_scores(base_scores, mode=normalize_mode)
        gap12 = math.inf
        if len(base_scores) >= 2:
            gap12 = base_scores[0] - base_scores[1]

        question = qid_to_question[qid]
        cand_texts = [make_candidate_text(c["text"], candidate_text_mode) for c in cands]

        if reranker_type == "bi_encoder":
            if bi_model is None:
                raise ValueError("bi_encoder model not provided")
            q_emb = bi_model.encode([question], convert_to_tensor=True, normalize_embeddings=True)
            cand_embs = bi_model.encode(cand_texts, convert_to_tensor=True, normalize_embeddings=True)
            sims = util.cos_sim(q_emb, cand_embs).cpu().numpy().reshape(-1)
            aux_raw = (sims + 1.0) / 2.0
        else:
            if cross_model is None:
                raise ValueError("cross_encoder model not provided")
            pairs = [(question, c_text) for c_text in cand_texts]
            preds = cross_model.predict(pairs, batch_size=batch_size)
            aux_raw = np.array(preds, dtype=np.float32)
            if aux_raw.ndim == 2:
                if aux_raw.shape[1] == 1:
                    aux_raw = aux_raw[:, 0]
                else:
                    logits = aux_raw - aux_raw.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)
                    aux_raw = probs[:, -1]
        aux_norm = normalize_scores(aux_raw, mode=normalize_mode)

        gold_norms = [normalize_answer(g) for g in qid_to_answers[qid]]
        base_top1_text = cands[0]["text"]
        base_correct = any(normalize_answer(base_top1_text) == g for g in gold_norms)

        entries.append({
            "qid": qid,
            "question": question,
            "cands": cands,
            "base_scores": base_scores,
            "base_norm": base_norm,
            "aux_norm": aux_norm,
            "gap12": gap12,
            "base_top1": base_top1_text,
            "base_correct": base_correct,
            "gold_norms": gold_norms,
        })

    if skipped_missing_question:
        print(f"[{label}] Skipped {skipped_missing_question} qids (missing question/answers)")
    return entries


def run_combination(entries, alpha: float, min_gap: float) -> Tuple[Dict[str, str], Dict[str, int]]:
    preds: Dict[str, str] = {}
    counts = {
        "total": 0,
        "unchanged": 0,
        "changed": 0,
        "changed_orig_incorrect_new_correct": 0,
        "changed_orig_correct_new_incorrect": 0,
        "changed_orig_incorrect_new_incorrect": 0,
        "changed_orig_correct_new_correct": 0,
    }

    for entry in entries:
        qid = entry["qid"]
        counts["total"] += 1
        cands = entry["cands"]
        if not cands:
            preds[qid] = ""
            counts["unchanged"] += 1
            continue

        selected_idx = 0
        if len(cands) >= 2 and entry["gap12"] < min_gap:
            combined = alpha * entry["base_norm"] + (1.0 - alpha) * entry["aux_norm"]
            selected_idx = int(np.argmax(combined))

        selected_text = cands[selected_idx]["text"]
        preds[qid] = selected_text

        base_text_norm = normalize_answer(entry["base_top1"])
        selected_norm = normalize_answer(selected_text)
        if selected_idx == 0 or selected_norm == base_text_norm:
            counts["unchanged"] += 1
            continue

        counts["changed"] += 1
        base_correct = entry["base_correct"]
        gold_norms = entry["gold_norms"]
        new_correct = any(selected_norm == g for g in gold_norms)
        if not base_correct and new_correct:
            counts["changed_orig_incorrect_new_correct"] += 1
        elif base_correct and not new_correct:
            counts["changed_orig_correct_new_incorrect"] += 1
        elif not base_correct and not new_correct:
            counts["changed_orig_incorrect_new_incorrect"] += 1
        else:
            counts["changed_orig_correct_new_correct"] += 1

    return preds, counts


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter search for thresholded semantic reranking")
    ap.add_argument("--dev_file", required=True, help="Path to dev-v1.1.json")
    ap.add_argument("--nbest", action="append", required=True, help="Label=path to nbest JSON (top-k candidates)")
    ap.add_argument("--alpha_start", type=float, default=0.2)
    ap.add_argument("--alpha_end", type=float, default=0.8)
    ap.add_argument("--alpha_step", type=float, default=0.1)
    ap.add_argument("--mingap_start", type=float, default=0.15)
    ap.add_argument("--mingap_end", type=float, default=0.35)
    ap.add_argument("--mingap_step", type=float, default=0.05)
    ap.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "softmax", "none"], help="Normalization for baseline scores")
    ap.add_argument("--candidate_text_mode", type=str, default="answer_is", choices=["answer_is", "raw"], help="Text template for candidate spans")
    ap.add_argument("--cap_topk", type=int, default=None, help="Cap candidates per question after cleanup")
    ap.add_argument("--max_answer_tokens", type=int, default=30)
    ap.add_argument("--reranker_type", type=str, default="cross_encoder", choices=["cross_encoder", "bi_encoder"])
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cross_model_name", type=str, default="cross-encoder/nli-deberta-base")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--results_csv", type=str, default=None, help="Optional path to write CSV summary")
    ap.add_argument("--results_table", type=str, default=None, help="Optional path to write plain text table in evaluation order")
    args = ap.parse_args()

    nbest_map = parse_label_path(args.nbest)
    dev_file = Path(args.dev_file)
    if not dev_file.exists():
        raise FileNotFoundError(dev_file)

    print("Loading dev dataset...")
    qid_to_question, qid_to_answers = load_dev_dataset(dev_file)
    print(f"Loaded {len(qid_to_question)} questions")

    alphas = []
    v = args.alpha_start
    while v <= args.alpha_end + 1e-9:
        alphas.append(round(v, 6))
        v += args.alpha_step
    min_gaps = []
    v = args.mingap_start
    while v <= args.mingap_end + 1e-9:
        min_gaps.append(round(v, 6))
        v += args.mingap_step

    print(f"Grid search: {len(alphas)} alpha values × {len(min_gaps)} min_gap values")

    if args.reranker_type == "bi_encoder":
        print(f"Loading bi-encoder model: {args.model_name}")
        bi_model = SentenceTransformer(args.model_name)
        cross_model = None
    else:
        print(f"Loading cross-encoder model: {args.cross_model_name}")
        bi_model = None
        cross_model = CrossEncoder(args.cross_model_name)

    results = []
    best_overall = None

    for label, path in nbest_map.items():
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"\n=== Preparing candidates for {label} ({path}) ===")
        entries = build_entries(
            label=label,
            nbest_path=path,
            qid_to_question=qid_to_question,
            qid_to_answers=qid_to_answers,
            normalize_mode=args.normalize,
            candidate_text_mode=args.candidate_text_mode,
            max_answer_tokens=args.max_answer_tokens,
            cap_topk=args.cap_topk or 0,
            reranker_type=args.reranker_type,
            batch_size=args.batch_size,
            bi_model=bi_model,
            cross_model=cross_model,
        )
        print(f"Prepared {len(entries)} entries")

        for alpha in alphas:
            for min_gap in min_gaps:
                start = time.time()
                preds, counts = run_combination(entries, alpha, min_gap)
                exact, f1 = compute_metrics(qid_to_answers, preds)
                duration = time.time() - start

                results.append({
                    "label": label,
                    "alpha": alpha,
                    "min_gap": min_gap,
                    "exact": exact,
                    "f1": f1,
                    "counts": counts,
                    "time_sec": duration,
                })

                if best_overall is None or f1 > best_overall["f1"]:
                    best_overall = {
                        "label": label,
                        "alpha": alpha,
                        "min_gap": min_gap,
                        "exact": exact,
                        "f1": f1,
                    }

                print(
                    f"[{label}] α={alpha:.2f}, gap<{min_gap:.2f} → exact={exact:6.2f} f1={f1:6.2f} "
                    f"changed={counts['changed']:5d} (+{counts['changed_orig_incorrect_new_correct']}/-{counts['changed_orig_correct_new_incorrect']}) "
                    f"best_f1={best_overall['f1']:6.2f} ({best_overall['label']}, α={best_overall['alpha']:.2f}, gap<{best_overall['min_gap']:.2f}) "
                    f"[{duration:.1f}s]"
                )

    header = f"{'Label':<12}{'Alpha':>7}{'MinGap':>8}{'Exact':>10}{'F1':>10}{'Changed':>10}"

    if args.results_table:
        table_path = Path(args.results_table)
        with table_path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for row in results:
                counts = row["counts"]
                f.write(
                    f"{row['label']:<12}{row['alpha']:>7.2f}{row['min_gap']:>8.2f}"
                    f"{row['exact']:>10.2f}{row['f1']:>10.2f}{counts['changed']:>10d}\n"
                )
        print(f"\nTable written to {table_path}")

    print("\n=== SUMMARY TABLE (sorted by F1 desc) ===")
    print(header)
    print("-" * len(header))
    for row in sorted(results, key=lambda r: r["f1"], reverse=True):
        counts = row["counts"]
        print(
            f"{row['label']:<12}{row['alpha']:>7.2f}{row['min_gap']:>8.2f}"
            f"{row['exact']:>10.2f}{row['f1']:>10.2f}{counts['changed']:>10d}"
        )

    if args.results_csv:
        import csv

        csv_path = Path(args.results_csv)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label",
                "alpha",
                "min_gap",
                "exact",
                "f1",
                "changed",
                "changed_orig_incorrect_new_correct",
                "changed_orig_correct_new_incorrect",
                "changed_orig_incorrect_new_incorrect",
                "changed_orig_correct_new_correct",
                "time_sec",
            ])
            for row in results:
                counts = row["counts"]
                writer.writerow([
                    row["label"],
                    row["alpha"],
                    row["min_gap"],
                    row["exact"],
                    row["f1"],
                    counts["changed"],
                    counts["changed_orig_incorrect_new_correct"],
                    counts["changed_orig_correct_new_incorrect"],
                    counts["changed_orig_incorrect_new_incorrect"],
                    counts["changed_orig_correct_new_correct"],
                    row["time_sec"],
                ])
        print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
