import importlib.util
from pathlib import Path

import pytest

from verifier import squad

ROOT = Path(__file__).resolve().parents[1]


def _official():
    spec = importlib.util.spec_from_file_location("official", ROOT / "src" / "evaluate-v2.0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("gold", "pred"),
    [
        ("Denver Broncos", "the Denver Broncos"),
        ("Denver Broncos", "Broncos"),
        ("1,000", "1000"),
        ("Saint Bernadette Soubirous", "Saint Bernadette"),
        ("an apple", "Apple!"),
        ("", "something"),
    ],
)
def test_metrics_match_the_official_script(gold: str, pred: str) -> None:
    official = _official()
    assert squad.exact_match(gold, pred) == official.compute_exact(gold, pred)
    assert squad.f1_score(gold, pred) == pytest.approx(official.compute_f1(gold, pred))


def test_split_is_deterministic_and_respects_shares() -> None:
    titles = [f"article_{i}" for i in range(4000)]
    first = [squad.assign_split(t) for t in titles]
    assert first == [squad.assign_split(t) for t in titles]
    share = {name: first.count(name) / len(titles) for name in squad.SPLITS}
    assert share["reader_train"] == pytest.approx(0.85, abs=0.02)
    assert share["reader_val"] == pytest.approx(0.05, abs=0.01)
    assert share["heldout"] == pytest.approx(0.10, abs=0.015)


def test_split_keeps_articles_whole() -> None:
    data = {
        "data": [
            {
                "title": f"t{i}",
                "paragraphs": [
                    {
                        "context": "c",
                        "qas": [
                            {
                                "id": f"q{i}",
                                "question": "?",
                                "answers": [{"text": "c", "answer_start": 0}],
                            }
                        ],
                    }
                ],
            }
            for i in range(50)
        ]
    }
    parts = squad.split_articles(data)
    assert sum(len(p["data"]) for p in parts.values()) == 50
    ids = [squad.question_ids(p) for p in parts.values()]
    assert sum(len(i) for i in ids) == 50
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_bad_shares_are_rejected() -> None:
    with pytest.raises(ValueError):
        squad.assign_split("x", {"reader_train": 0.5, "reader_val": 0.2, "heldout": 0.2})
