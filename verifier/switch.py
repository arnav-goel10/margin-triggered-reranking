"""Switch rules: keep the reader's first answer, or take its second.

Switching only changes exact match on informative questions (exactly one of
the two candidates is right): it fixes one where candidate 2 is right and
breaks one where candidate 1 is right. Every rule is scored by that net count.

Rules, each tuned on the held-out slice of SQuAD train only:

- ``reader``: never switch (the baseline).
- ``verifier``: switch when the verifier prefers candidate 2 by more than
  ``delta`` logits, whatever the reader's gap.
- ``gated``: as ``verifier``, but only when the reader's top-two gap is at most
  ``tau``. This is the margin trigger with a trained second opinion.
- ``learned``: logistic regression on the features in ``verifier.features``,
  switching when its probability that candidate 2 is the right one exceeds a
  threshold chosen on the training folds.
- ``oracle``: switch exactly when it helps (an upper bound, not a method).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from verifier.features import TopTwo, feature_matrix


@dataclass(frozen=True)
class Outcome:
    switched: int
    fixed: int
    broken: int
    questions: int

    @property
    def net(self) -> int:
        return self.fixed - self.broken

    @property
    def em_delta(self) -> float:
        """Change in exact match, in points."""
        return 100 * self.net / self.questions


def outcome(records: Sequence[TopTwo], switch: Sequence[bool]) -> Outcome:
    if len(records) != len(switch):
        raise ValueError("one decision per record")
    if not records:
        raise ValueError("no records")
    fixed = sum(
        1 for r, s in zip(records, switch, strict=True) if s and r.c2_correct and not r.c1_correct
    )
    broken = sum(
        1 for r, s in zip(records, switch, strict=True) if s and r.c1_correct and not r.c2_correct
    )
    return Outcome(sum(map(bool, switch)), fixed, broken, len(records))


@dataclass(frozen=True)
class ThresholdRule:
    """Switch when gap <= tau and v2 - v1 > delta. tau = inf drops the gate."""

    tau: float
    delta: float

    def decide(self, records: Sequence[TopTwo]) -> list[bool]:
        return [r.gap <= self.tau and (r.v2 - r.v1) > self.delta for r in records]


def tune_threshold_rule(
    records: Sequence[TopTwo], taus: Sequence[float], deltas: Sequence[float]
) -> tuple[ThresholdRule, Outcome]:
    """Grid search for the best net gain. Ties go to the rule that switches less."""
    best: tuple[ThresholdRule, Outcome] | None = None
    for tau in taus:
        for delta in deltas:
            rule = ThresholdRule(tau, delta)
            result = outcome(records, rule.decide(records))
            if best is None or (result.net, -result.switched) > (
                best[1].net,
                -best[1].switched,
            ):
                best = (rule, result)
    if best is None:
        raise ValueError("empty grid")
    return best


def oracle(records: Sequence[TopTwo]) -> list[bool]:
    return [r.c2_correct and not r.c1_correct for r in records]


def fit_logistic(
    x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iterations: int = 50
) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method.

    ``x`` must already include a bias column. The bias is not penalised.
    """
    n, d = x.shape
    if y.shape != (n,):
        raise ValueError("y must have one label per row")
    w = np.zeros(d)
    penalty = np.full(d, l2)
    penalty[0] = 0.0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-(x @ w)))
        gradient = x.T @ (p - y) + penalty * w
        hessian = (x.T * (p * (1 - p))) @ x + np.diag(penalty) + 1e-9 * np.eye(d)
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


@dataclass(frozen=True)
class LearnedSwitch:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    threshold: float

    def probability(self, records: Sequence[TopTwo]) -> np.ndarray:
        z = (feature_matrix(records) - self.mean) / self.scale
        x = np.hstack([np.ones((len(records), 1)), z])
        return np.asarray(1 / (1 + np.exp(-(x @ self.weights))))

    def decide(self, records: Sequence[TopTwo]) -> list[bool]:
        return [bool(p > self.threshold) for p in self.probability(records)]


def fit_learned_switch(
    records: Sequence[TopTwo],
    *,
    l2: float = 1.0,
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.30, 0.951, 0.025), 3)),
) -> LearnedSwitch:
    """Fit on informative records, then pick the threshold with the best net gain.

    Only informative records carry a label that matters (is candidate 2 the
    right one?). The threshold is chosen on all of ``records``, ties going to
    the higher threshold, which switches less.
    """
    train = [r for r in records if r.informative]
    if len(train) < 10:
        raise ValueError("too few informative records to fit a switch")
    features = feature_matrix(train)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    x = np.hstack([np.ones((len(train), 1)), (features - mean) / scale])
    y = np.array([float(r.c2_correct) for r in train])
    weights = fit_logistic(x, y, l2=l2)
    probe = LearnedSwitch(mean, scale, weights, threshold=1.0)
    probs = probe.probability(records)
    best_t, best_key = 1.0, (0, 0)
    for t in sorted(thresholds, reverse=True):
        result = outcome(records, [bool(p > t) for p in probs])
        key = (result.net, -result.switched)
        if key > best_key:
            best_t, best_key = float(t), key
    return LearnedSwitch(mean, scale, weights, best_t)


def folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    """Shuffled index folds of near-equal size."""
    if not 2 <= k <= n:
        raise ValueError("need 2 <= k <= n")
    order = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(order, k)]


def paired_bootstrap_em(
    baseline_em: Sequence[int],
    system_em: Sequence[int],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """EM difference in points, system minus baseline, with a 95% interval."""
    a = np.asarray(baseline_em, dtype=float)
    b = np.asarray(system_em, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired 1-d arrays of equal, non-zero length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    draws = 100 * (b[idx].mean(axis=1) - a[idx].mean(axis=1))
    return {
        "estimate": float(100 * (b.mean() - a.mean())),
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "p_not_better": float(np.mean(draws <= 0)),
    }
