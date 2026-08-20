#!/usr/bin/env python3
# Rerank SQuAD candidates only when the top-two baseline scores are close
import argparse
import json
from typing import Dict

from sentence_transformers import SentenceTransformer, CrossEncoder

from rerank_squad_candidates import (
    load_dev_qid_to_question,
    load_nbest,
    dedup_and_filter_candidates,
    rerank_for_qid,
)


def main():
    ap = argparse.ArgumentParser(description="Thresholded semantic reranking for SQuAD n-best candidates")
    ap.add_argument("--dev_file", type=str, required=True, help="Path to dev-v1.1.json")
    ap.add_argument("--nbest_file", type=str, required=True, help="Path to predictions_with_top_k.json (qid -> list of candidates)")
    ap.add_argument("--out_file", type=str, default="predictions_with_top_k_reranked.json", help="Output predictions file (qid -> best span text)")
    ap.add_argument("--alpha", type=float, default=0.75, help="Fusion weight for baseline score (0..1)")
    ap.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "softmax", "none"], help="Per-question normalization for baseline scores")
    ap.add_argument("--candidate_text_mode", type=str, default="answer_is", choices=["answer_is", "raw"], help="How to embed candidate spans")
    ap.add_argument("--cap_topk", type=int, default=20, help="Max candidates to consider per question after cleanup")
    ap.add_argument("--max_answer_tokens", type=int, default=30, help="Drop candidate spans longer than N tokens")
    ap.add_argument("--reranker_type", type=str, default="cross_encoder", choices=["bi_encoder", "cross_encoder"], help="Which semantic reranker to apply")
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model for embeddings (bi-encoder mode)")
    ap.add_argument("--cross_model_name", type=str, default="cross-encoder/nli-deberta-base", help="Cross-encoder model to use when reranker_type=cross_encoder")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for encoder inference")
    ap.add_argument("--min_gap", type=float, default=0.35, help="Only rerank when (score1 - score2) < min_gap")
    args = ap.parse_args()

    print("Loading dev questions...")
    qid2q = load_dev_qid_to_question(args.dev_file)
    print(f"Loaded {len(qid2q)} questions")

    print("Loading n-best candidates...")
    qid2cands_all = load_nbest(args.nbest_file)
    print(f"Loaded candidates for {len(qid2cands_all)} qids")

    qids = [qid for qid in qid2cands_all.keys() if qid in qid2q]

    if args.reranker_type == "bi_encoder":
        print("Loading bi-encoder model:", args.model_name)
        bi_model = SentenceTransformer(args.model_name)
        cross_model = None
    else:
        print("Loading cross-encoder model:", args.cross_model_name)
        bi_model = None
        cross_model = CrossEncoder(args.cross_model_name)

    outputs: Dict[str, str] = {}
    total_considered = 0
    reranked = 0
    skipped_due_gap = 0
    insufficient_candidates = 0

    for idx, qid in enumerate(qids, 1):
        question = qid2q[qid]
        cands_raw = qid2cands_all.get(qid, [])
        cands = dedup_and_filter_candidates(cands_raw, max_tokens=args.max_answer_tokens)
        if args.cap_topk and len(cands) > args.cap_topk:
            cands = cands[: args.cap_topk]

        if not cands:
            outputs[qid] = ""
            insufficient_candidates += 1
        else:
            # Check gap AFTER dedup/filter (matching search script behavior)
            should_rerank = False
            if len(cands) >= 2:
                score1 = float(cands[0].get("score", 0.0))
                score2 = float(cands[1].get("score", 0.0))
                gap = score1 - score2
                if gap < args.min_gap:
                    should_rerank = True
                else:
                    skipped_due_gap += 1
            else:
                insufficient_candidates += 1

            if should_rerank:
                best_text = rerank_for_qid(
                    question=question,
                    candidates=cands,
                    alpha=args.alpha,
                    norm_mode=args.normalize,
                    cand_text_mode=args.candidate_text_mode,
                    reranker_type=args.reranker_type,
                    bi_encoder=bi_model,
                    cross_encoder=cross_model,
                    batch_size=args.batch_size,
                )
                reranked += 1
                outputs[qid] = best_text
            else:
                outputs[qid] = cands[0]["text"]

        total_considered += len(cands)
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(qids)} qids")

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False)

    print("Done")
    print(f"QIDs processed: {len(qids)}")
    print(f"Avg candidates considered per QID: {total_considered / max(len(qids),1):.2f}")
    print(f"QIDs reranked: {reranked}")
    print(f"QIDs skipped due to confident gap: {skipped_due_gap}")
    print(f"QIDs without >=2 candidates: {insufficient_candidates}")
    print(f"Wrote: {args.out_file}")


if __name__ == "__main__":
    main()
