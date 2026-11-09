#!/usr/bin/env python3
"""Split SQuAD v1.1 train by article into reader_train / reader_val / heldout.

Writes SQuAD-format files under data/splits/ (not committed: they are SQuAD
text) and a summary with counts and id digests to results/verifier/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verifier import squad  # noqa: E402
from verifier.protocol import assert_disjoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train-v1.1.json")
    parser.add_argument("--dev", type=Path, default=ROOT / "data" / "dev-v1.1.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "results" / "verifier" / "split_summary.json"
    )
    args = parser.parse_args()

    train = squad.load(args.train)
    parts = squad.split_articles(train)
    ids = {name: squad.question_ids(part) for name, part in parts.items()}
    dev_ids = squad.question_ids(squad.load(args.dev))
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert_disjoint(a, ids[a], b, ids[b])
        assert_disjoint(a, ids[a], "dev", dev_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"salt": squad.SALT, "shares": squad.DEFAULT_SHARES}
    for name, part in parts.items():
        (args.out_dir / f"{name}.json").write_text(json.dumps(part), encoding="utf-8")
        digest = hashlib.sha256("\n".join(sorted(ids[name])).encode()).hexdigest()
        summary[name] = {
            "articles": len(part["data"]),
            "questions": len(ids[name]),
            "qid_sha256": digest,
        }
        print(f"{name:13s} {len(part['data']):4d} articles {len(ids[name]):6d} questions")
    summary["dev"] = {"questions": len(dev_ids)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
