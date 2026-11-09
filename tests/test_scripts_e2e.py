"""Run the CPU stages of the pipeline end to end on synthetic SQuAD files."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _world(prefix: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    articles, candidates, logits = [], {}, {}
    for i in range(n):
        qid = f"{prefix}{i}"
        gap = float(rng.uniform(0, 0.8))
        c2_right = rng.uniform() < max(0.0, 0.5 - gap)
        context = "alpha beta gamma delta"
        gold = "gamma" if c2_right else "alpha"
        articles.append(
            {
                "title": qid,
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": qid,
                                "question": "which?",
                                "answers": [{"text": gold, "answer_start": context.index(gold)}],
                            }
                        ],
                    }
                ],
            }
        )
        candidates[qid] = [
            {"text": "alpha", "score": 0.9, "start": 0, "end": 5},
            {"text": "gamma", "score": 0.9 - gap, "start": 11, "end": 16},
        ]
        logits[qid] = [0.0, float(rng.normal(2.0 if c2_right else -1.0, 1.0))]
    return {"version": "1.1", "data": articles}, candidates, logits


def _write(tmp: Path, name: str, world) -> tuple[Path, Path, Path]:
    squad_file, cands, logs = tmp / f"{name}.json", tmp / f"{name}_c.json", tmp / f"{name}_l.json"
    for path, obj in zip((squad_file, cands, logs), world, strict=True):
        path.write_text(json.dumps(obj))
    return squad_file, cands, logs


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)


def test_fit_then_single_dev_read(tmp_path: Path) -> None:
    h_squad, h_c, h_l = _write(tmp_path, "heldout", _world("h", 400, 0))
    d_squad, d_c, d_l = _write(tmp_path, "dev", _world("d", 400, 1))
    report = tmp_path / "heldout_report.json"
    fit = _run(
        "scripts/verifier/fit_switch.py",
        "--heldout",
        str(h_squad),
        "--candidates",
        str(h_c),
        "--logits",
        str(h_l),
        "--dev",
        str(d_squad),
        "--out",
        str(report),
    )
    assert fit.returncode == 0, fit.stderr
    assert "selected for dev" in fit.stdout

    dev_report = tmp_path / "dev_report.json"
    args = (
        "scripts/verifier/final_eval.py",
        "--dev",
        str(d_squad),
        "--heldout-report",
        str(report),
        "--candidates",
        str(d_c),
        "--logits",
        str(d_l),
        "--out",
        str(dev_report),
    )
    refused = _run(*args)
    assert refused.returncode != 0 and "--final" in refused.stderr

    first = _run(*args, "--final")
    assert first.returncode == 0, first.stderr
    systems = json.loads(dev_report.read_text())["systems"]
    assert systems["gated"]["em_delta_vs_reader"]["estimate"] > 0
    assert systems["reader"]["switched"] == 0

    second = _run(*args, "--final")
    assert second.returncode != 0 and "already been scored" in second.stderr


def test_fit_refuses_heldout_that_overlaps_dev(tmp_path: Path) -> None:
    world = _world("x", 50, 0)
    squad_file, cands, logs = _write(tmp_path, "heldout", world)
    fit = _run(
        "scripts/verifier/fit_switch.py",
        "--heldout",
        str(squad_file),
        "--candidates",
        str(cands),
        "--logits",
        str(logs),
        "--dev",
        str(squad_file),
        "--out",
        str(tmp_path / "r.json"),
    )
    assert fit.returncode != 0 and "in both heldout and dev" in fit.stderr
