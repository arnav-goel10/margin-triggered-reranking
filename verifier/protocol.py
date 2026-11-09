"""Guards for the evaluation protocol.

Everything is tuned on the held-out slice of SQuAD train. The dev set is read
once, by ``scripts/verifier/final_eval.py --final``, which refuses to run if
its report already exists. Rerunning on dev means deleting the report by hand,
which shows up in version control.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class ProtocolError(RuntimeError):
    pass


def assert_disjoint(name_a: str, ids_a: Iterable[str], name_b: str, ids_b: Iterable[str]) -> None:
    overlap = set(ids_a) & set(ids_b)
    if overlap:
        raise ProtocolError(
            f"{len(overlap)} question ids are in both {name_a} and {name_b}, "
            f"for example {sorted(overlap)[:3]}"
        )


def claim_final_run(report: Path, *, final: bool) -> None:
    """Allow one dev-set evaluation, and only when asked for explicitly."""
    if not final:
        raise ProtocolError("the dev set is read only with --final")
    if report.exists():
        raise ProtocolError(
            f"{report} exists: the dev set has already been scored once. "
            "Delete it and commit the deletion to score again."
        )
