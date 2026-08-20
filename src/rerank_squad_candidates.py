import argparse, json, math, os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util, CrossEncoder


def load_dev_qid_to_question(dev_path: str) -> Dict[str, str]:
    # Build qid -> question text mapping from SQuAD v1.1 dev file
    with open(dev_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    qid2q = {}
    for art in js["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                qid2q[qa["id"]] = qa["question"]
    return qid2q


def load_nbest(nbest_path: str) -> Dict[str, List[dict]]:
    # Load top-k candidates per qid: { qid: [ {text, score, start, end}, ... ] }
    with open(nbest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dedup_and_filter_candidates(cands: List[dict], max_tokens: int = 30) -> List[dict]:
    # Strip whitespace, drop empties, deduplicate by text, optionally drop very long spans
    seen = set()
    cleaned = []
    for c in cands:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        # Approx token count by whitespace split
        if max_tokens is not None and len(text.split()) > max_tokens:
            continue
        if text in seen:
            # keep only higher-scoring duplicate
            for i, cc in enumerate(cleaned):
                if cc["text"] == text and float(c.get("score", 0.0)) > cc["score"]:
                    cleaned[i] = {"text": text, "score": float(c.get("score", 0.0))}
                    break
            continue
        cleaned.append({"text": text, "score": float(c.get("score", 0.0))})
        seen.add(text)
    return cleaned


def normalize_scores(scores: np.ndarray, mode: str = "minmax") -> np.ndarray:
    # Normalize baseline candidate scores per question
    if scores.size == 0:
        return scores
    if mode == "none":
        return scores
    if mode == "softmax":
        # softmax normalization
        x = scores - scores.max()
        exps = np.exp(x)
        denom = exps.sum()
        return exps / max(denom, 1e-12)
    # default: minmax
    lo, hi = scores.min(), scores.max()
    if math.isclose(hi, lo):
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo + 1e-12)


def make_candidate_text(text: str, mode: str) -> str:
    if mode == "answer_is":
        return f"The answer is {text}"
    return text


def rerank_for_qid(
    question: str,
    candidates: List[dict],
    alpha: float,
    norm_mode: str,
    cand_text_mode: str,
    reranker_type: str,
    bi_encoder: SentenceTransformer = None,
    cross_encoder: CrossEncoder = None,
    batch_size: int = 32,
) -> str:
    # Return best candidate text after fusing baseline and semantic scores
    if not candidates:
        return ""

    q_text = question.strip()
    cand_texts = [make_candidate_text(c["text"], cand_text_mode) for c in candidates]

    base_scores = np.array([float(c["score"]) for c in candidates], dtype=np.float32)
    base_norm = normalize_scores(base_scores, mode=norm_mode)

    if reranker_type == "bi_encoder":
        if bi_encoder is None:
            raise ValueError("bi_encoder model must be provided for bi_encoder reranking")
        q_emb = bi_encoder.encode([q_text], convert_to_tensor=True, normalize_embeddings=True)
        cand_embs = bi_encoder.encode(cand_texts, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(q_emb, cand_embs).cpu().numpy().reshape(-1)
        aux_scores = (sims + 1.0) / 2.0  # map cosine similarity to [0,1]
        aux_scores = normalize_scores(aux_scores, mode=norm_mode)
    elif reranker_type == "cross_encoder":
        if cross_encoder is None:
            raise ValueError("cross_encoder model must be provided for cross_encoder reranking")
        pairs = [(q_text, c_text) for c_text in cand_texts]
        preds = cross_encoder.predict(pairs, batch_size=batch_size)
        aux_scores = np.array(preds, dtype=np.float32)
        if aux_scores.ndim == 2:
            if aux_scores.shape[1] == 1:
                aux_scores = aux_scores[:, 0]
            else:
                # Assume final column is entailment class, convert logits to probs
                logits = aux_scores
                logits = logits - logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)
                aux_scores = probs[:, -1]
        aux_scores = normalize_scores(aux_scores, mode=norm_mode)
    else:
        raise ValueError(f"Unknown reranker_type: {reranker_type}")

    final = alpha * base_norm + (1.0 - alpha) * aux_scores
    best_idx = int(final.argmax())
    return candidates[best_idx]["text"]


def main():
    ap = argparse.ArgumentParser(description="Semantic re-ranking for SQuAD n-best candidates")
    ap.add_argument("--dev_file", type=str, required=True, help="Path to dev-v1.1.json")
    ap.add_argument("--nbest_file", type=str, required=True, help="Path to predictions_with_5_acc2.json (qid -> list of candidates)")
    ap.add_argument("--out_file", type=str, default="predictions_reranked.json", help="Output predictions file (qid -> best span text)")
    ap.add_argument("--alpha", type=float, default=0.7, help="Fusion weight for baseline score (0..1)")
    ap.add_argument("--normalize", type=str, default="minmax", choices=["minmax", "softmax", "none"], help="Per-question normalization for baseline scores")
    ap.add_argument("--candidate_text_mode", type=str, default="answer_is", choices=["answer_is", "raw"], help="How to embed candidate spans")
    ap.add_argument("--cap_topk", type=int, default=20, help="Max candidates to consider per question")
    ap.add_argument("--max_answer_tokens", type=int, default=30, help="Drop candidate spans longer than N tokens")
    ap.add_argument("--reranker_type", type=str, default="cross_encoder", choices=["bi_encoder", "cross_encoder"], help="Which semantic reranker to apply")
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model for embeddings (bi-encoder mode)")
    ap.add_argument("--cross_model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder model to use when reranker_type=cross_encoder")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for encoder inference")
    args = ap.parse_args()

    print("Loading dev questions...")
    qid2q = load_dev_qid_to_question(args.dev_file)
    print(f"Loaded {len(qid2q)} questions")

    print("Loading n-best candidates...")
    qid2cands_all = load_nbest(args.nbest_file)
    print(f"Loaded candidates for {len(qid2cands_all)} qids")

    # Only process qids that have both candidates and questions
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
    empty_or_missing = 0
    total_considered = 0

    for idx, qid in enumerate(qids, 1):
        question = qid2q[qid]
        cands_raw = qid2cands_all.get(qid, [])
        # Clean and filter candidates
        cands = dedup_and_filter_candidates(cands_raw, max_tokens=args.max_answer_tokens)
        if args.cap_topk and len(cands) > args.cap_topk:
            cands = cands[: args.cap_topk]

        if not cands:
            outputs[qid] = ""
            empty_or_missing += 1
        else:
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
            outputs[qid] = best_text

        total_considered += len(cands)
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(qids)} qids")

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False)

    print("Done")
    print(f"Processed {len(qids)} QIDs")
    print(f"Avg candidates per QID: {total_considered / max(len(qids),1):.2f}")
    print(f"Empty/missing: {empty_or_missing}")
    print(f"Output: {args.out_file}")
    print(f"\nEvaluate with: python evaluate-v2.0.py {args.dev_file} {args.out_file}")


if __name__ == "__main__":
    main()
