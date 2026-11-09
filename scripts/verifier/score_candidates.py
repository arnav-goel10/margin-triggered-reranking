#!/usr/bin/env python3
"""Verifier logits for the top-k reader candidates of every question.

Writes {qid: [logit_rank1, logit_rank2, ...]} where each logit is the
verifier's log-odds that the span is the answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verifier import squad  # noqa: E402
from verifier.pairs import scoring_inputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--squad-file", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; refusing to run on CPU")
    tokenizer = AutoTokenizer.from_pretrained(args.verifier)
    model = AutoModelForSequenceClassification.from_pretrained(args.verifier).cuda().eval()
    questions = list(squad.iter_questions(squad.load(args.squad_file)))
    rows = scoring_inputs(questions, json.loads(args.candidates.read_text()), top_k=args.top_k)
    print(f"{len(rows)} candidate spans over {len(questions)} questions")
    logits: dict[str, list[float]] = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i : i + args.batch_size]
            enc = tokenizer(
                [r[2] for r in chunk],
                [r[3] for r in chunk],
                truncation="only_second",
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            out = model(**enc).logits.float()
            margin = (out[:, 1] - out[:, 0]).tolist()
            for (qid, rank, _, _), value in zip(chunk, margin, strict=True):
                scores = logits.setdefault(qid, [])
                if len(scores) != rank:
                    raise RuntimeError(f"rank order broken for {qid}")
                scores.append(round(float(value), 5))
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(logits), encoding="utf-8")
    print(f"wrote {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
