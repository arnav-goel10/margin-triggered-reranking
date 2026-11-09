#!/usr/bin/env python3
"""Top-k reader candidates for every question in a SQuAD-format file.

Same output format as src/evaluate_k_candidates.py ({qid: [{text, score,
start, end}, ...]}), with batched pipeline calls so 73k training questions
take minutes rather than hours. Pipeline defaults (max_answer_len 15, no
impossible answers) match the original script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verifier import squad  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--squad-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import pipeline

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; refusing to run on CPU")
    qa = pipeline("question-answering", model=args.model, tokenizer=args.model, device=0)
    questions = list(squad.iter_questions(squad.load(args.squad_file)))
    print(f"{len(questions)} questions from {args.squad_file}")
    started = time.time()
    outputs = qa(
        question=[q.question for q in questions],
        context=[q.context for q in questions],
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    predictions = {}
    for q, result in zip(questions, outputs, strict=True):
        if isinstance(result, dict):
            result = [result]
        predictions[q.qid] = [
            {
                "text": r["answer"],
                "score": float(r["score"]),
                "start": int(r["start"]),
                "end": int(r["end"]),
            }
            for r in result[: args.top_k]
        ]
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"wrote {args.out_file} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
