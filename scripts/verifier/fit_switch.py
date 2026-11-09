#!/usr/bin/env python3
"""Tune every switch rule on the held-out slice of SQuAD train (CPU, seconds).

Writes results/verifier/heldout_report.json: cross-validated fixed / broken /
EM delta per rule, the gap diagnostics, which rule is selected for the dev
read, and the fitted parameters final_eval.py replays on dev.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verifier import squad  # noqa: E402
from verifier.experiment import diagnostics, tune  # noqa: E402
from verifier.features import build_records  # noqa: E402
from verifier.protocol import assert_disjoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--dev", type=Path, default=ROOT / "data" / "dev-v1.1.json")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "verifier" / "heldout_report.json"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    heldout = squad.load(args.heldout)
    # Only dev ids are read here, never dev answers, to prove the sets are disjoint.
    assert_disjoint(
        "heldout", squad.question_ids(heldout), "dev", squad.question_ids(squad.load(args.dev))
    )
    questions = list(squad.iter_questions(heldout))
    records = build_records(
        questions, json.loads(args.candidates.read_text()), json.loads(args.logits.read_text())
    )
    report = tune(records, k=args.folds, seed=args.seed)
    report["diagnostics"] = diagnostics(records)
    reader = squad.score_predictions(questions, {r.qid: r.text1 for r in records})
    report["reader_heldout"] = {k: round(v, 3) for k, v in reader.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"reader on heldout: EM {reader['exact']:.3f} F1 {reader['f1']:.3f}")
    for rule, row in report["cross_validated"].items():
        print(
            f"{rule:9s} CV: switched {row['switched']:5d} fixed {row['fixed']:4d} "
            f"broken {row['broken']:4d} EM delta {row['em_delta']:+.3f}"
        )
    print(f"selected for dev: {report['selected']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
