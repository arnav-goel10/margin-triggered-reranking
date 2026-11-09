import math

import numpy as np
import pytest

from verifier.experiment import apply_rules, diagnostics, tune
from verifier.features import FEATURE_NAMES, TopTwo, build_records, feature_matrix
from verifier.protocol import ProtocolError, assert_disjoint, claim_final_run
from verifier.squad import Question
from verifier.switch import (
    ThresholdRule,
    fit_logistic,
    folds,
    oracle,
    outcome,
    paired_bootstrap_em,
    tune_threshold_rule,
)


def rec(qid, gap, dv, c1, c2, p1=0.6):
    return TopTwo(qid, "a", "b", p1, p1 - gap, 0.0, dv, c1, c2)


def test_outcome_counts_only_informative_switches() -> None:
    records = [
        rec("fix", 0.1, 1, False, True),
        rec("break", 0.1, 1, True, False),
        rec("both_right", 0.1, 1, True, True),
        rec("both_wrong", 0.1, 1, False, False),
    ]
    result = outcome(records, [True] * 4)
    assert (result.switched, result.fixed, result.broken, result.net) == (4, 1, 1, 0)
    assert outcome(records, oracle(records)).net == 1


def test_threshold_rule_needs_both_gate_and_preference() -> None:
    rule = ThresholdRule(tau=0.1, delta=0.5)
    records = [
        rec("a", 0.05, 1.0, False, True),
        rec("b", 0.2, 1.0, False, True),
        rec("c", 0.05, 0.2, False, True),
    ]
    assert rule.decide(records) == [True, False, False]
    assert ThresholdRule(math.inf, 0.5).decide(records) == [True, True, False]


def test_grid_prefers_the_rule_that_switches_less_on_ties() -> None:
    records = [rec("a", 0.05, 3.0, False, True), rec("b", 0.3, 0.1, True, True)]
    rule, result = tune_threshold_rule(records, (0.1, 1.0), (0.0, 1.0))
    assert result.net == 1
    assert result.switched == 1
    assert rule.decide(records) == [True, False]


def test_logistic_recovers_the_sign_of_a_clear_signal() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    y = (z + 0.3 * rng.normal(size=400) > 0).astype(float)
    w = fit_logistic(np.column_stack([np.ones(400), z]), y, l2=0.1)
    assert w[1] > 2


def test_folds_partition_the_indices() -> None:
    parts = folds(23, 5, seed=1)
    assert sorted(np.concatenate(parts).tolist()) == list(range(23))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def _synthetic(n=600, seed=0):
    """Verifier helps when it prefers candidate 2 strongly and the gap is small."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        gap = float(rng.uniform(0, 0.8))
        c2_better = rng.uniform() < max(0.0, 0.5 - gap)
        dv = float(rng.normal(2.0 if c2_better else -1.0, 1.0))
        records.append(rec(f"q{i}", gap, dv, not c2_better, c2_better, p1=0.9))
    return records


def test_tuning_finds_a_positive_rule_and_replays_it() -> None:
    records = _synthetic()
    report = tune(records, k=5, seed=0)
    cv = report["cross_validated"]
    assert cv["reader"]["net"] == 0
    assert cv["oracle"]["net"] >= cv["gated"]["net"] > 0
    assert report["selected"] in ("verifier", "gated", "learned")
    replay = apply_rules(_synthetic(seed=1), report["params"])
    assert outcome(_synthetic(seed=1), replay["gated"]).net > 0
    assert set(replay) == {"reader", "verifier", "gated", "learned", "oracle"}


def test_diagnostics_auroc_is_high_when_small_gaps_mean_errors() -> None:
    records = [rec(f"r{i}", 0.5 + i / 100, 0, True, False) for i in range(20)]
    records += [rec(f"w{i}", i / 100, 0, False, True) for i in range(20)]
    assert diagnostics(records)["gap_auroc_for_reader_error"] == pytest.approx(1.0)


def test_paired_bootstrap_em_reports_points() -> None:
    base = [1, 0, 0, 1] * 50
    better = [1, 1, 0, 1] * 50
    d = paired_bootstrap_em(base, better, resamples=200)
    assert d["estimate"] == pytest.approx(25.0)
    assert d["low"] > 0 and d["p_not_better"] == 0.0


def test_records_join_candidates_logits_and_gold() -> None:
    q = Question("q", "t", "ctx", "?", ("Denver Broncos", "Broncos"), (0, 7))
    cands = {
        "q": [{"text": "the Denver Broncos", "score": 0.5}, {"text": "Panthers", "score": 0.3}]
    }
    [r] = build_records([q], cands, {"q": [1.0, -2.0]})
    assert r.c1_correct and not r.c2_correct
    assert r.gap == pytest.approx(0.2)
    assert feature_matrix([r]).shape == (1, len(FEATURE_NAMES))


def test_single_candidate_never_switches() -> None:
    q = Question("q", "t", "ctx", "?", ("x",), (0,))
    [r] = build_records([q], {"q": [{"text": "y", "score": 0.9}]}, {"q": [0.0]})
    assert r.gap == 0 and r.v2 == r.v1


def test_protocol_guards(tmp_path) -> None:
    with pytest.raises(ProtocolError):
        assert_disjoint("a", {"1", "2"}, "b", {"2"})
    report = tmp_path / "dev_report.json"
    with pytest.raises(ProtocolError, match="--final"):
        claim_final_run(report, final=False)
    claim_final_run(report, final=True)
    report.write_text("{}")
    with pytest.raises(ProtocolError, match="already been scored"):
        claim_final_run(report, final=True)
