#!/usr/bin/env python3
"""The single dev-set read: replay the rules fitted on held-out data.

Nothing is tuned here. Every rule from the held-out report is applied as
fitted and reported, with the selected one marked, so the dev numbers cannot
be cherry-picked after the fact. Refuses to run without --final, or if the
report already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verifier import squad  # noqa: E402
from verifier.experiment import apply_rules  # noqa: E402
from verifier.features import build_records  # noqa: E402
from verifier.protocol import claim_final_run  # noqa: E402
from verifier.switch import outcome, paired_bootstrap_em  # noqa: E402


def evaluate(questions, candidates, logits, params):
    records = build_records(questions, candidates, logits)
    by_qid = {q.qid: q for q in questions}
    base_em = [squad.best_over_golds(by_qid[r.qid].answers, r.text1)[0] for r in records]
    rows = {}
    for rule, decisions in apply_rules(records, params).items():
        preds = {
            r.qid: (r.text2 if s else r.text1) for r, s in zip(records, decisions, strict=True)
        }
        em = [squad.best_over_golds(by_qid[r.qid].answers, preds[r.qid])[0] for r in records]
        result = outcome(records, decisions)
        scores = squad.score_predictions(questions, preds)
        rows[rule] = {
            "exact": round(scores["exact"], 3),
            "f1": round(scores["f1"], 3),
            "switched": result.switched,
            "fixed": result.fixed,
            "broken": result.broken,
            "em_delta_vs_reader": {
                k: round(v, 4) for k, v in paired_bootstrap_em(base_em, em).items()
            },
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--dev", type=Path, default=ROOT / "data" / "dev-v1.1.json")
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument(
        "--published-candidates",
        type=Path,
        default=None,
        help="also replay the rules on the published reader's dev candidates",
    )
    parser.add_argument("--published-logits", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "verifier" / "dev_report.json"
    )
    args = parser.parse_args()

    claim_final_run(args.out, final=args.final)
    heldout = json.loads(args.heldout_report.read_text())
    questions = list(squad.iter_questions(squad.load(args.dev)))
    report = {
        "split": "SQuAD v1.1 dev",
        "questions": len(questions),
        "selected_on_heldout": heldout["selected"],
        "reader_trained_on": "reader_train (85% of SQuAD train articles)",
        "systems": evaluate(
            questions,
            json.loads(args.candidates.read_text()),
            json.loads(args.logits.read_text()),
            heldout["params"],
        ),
    }
    if args.published_candidates and args.published_logits:
        report["published_reader"] = evaluate(
            questions,
            json.loads(args.published_candidates.read_text()),
            json.loads(args.published_logits.read_text()),
            heldout["params"],
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ("systems", "published_reader"):
        for rule, row in report.get(name, {}).items():
            d = row["em_delta_vs_reader"]
            mark = " (selected)" if rule == report["selected_on_heldout"] else ""
            print(
                f"{name:16s} {rule:9s} EM {row['exact']:.3f} F1 {row['f1']:.3f} "
                f"fixed {row['fixed']:4d} broken {row['broken']:4d} "
                f"dEM {d['estimate']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]{mark}"
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
