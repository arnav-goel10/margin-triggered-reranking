import pytest

from verifier.pairs import END, START, locate, mark_span, scoring_inputs, training_pairs
from verifier.squad import Question

CONTEXT = "The Broncos beat the Panthers 24 to 10 in Super Bowl 50."


def _q(answers=("Broncos",), starts=(4,)):
    return Question("q1", "t", CONTEXT, "Who won?", answers, starts)


def test_mark_span_wraps_exactly_the_span() -> None:
    marked = mark_span(CONTEXT, 4, 11)
    assert f"{START} Broncos {END}" in marked
    assert marked.replace(f"{START} ", "").replace(f" {END}", "") == CONTEXT


def test_mark_span_windows_long_contexts() -> None:
    context = "x " * 1000 + "answer" + " y" * 1000
    start = context.index("answer")
    marked = mark_span(context, start, start + 6, window=20)
    assert marked.startswith("... ") and marked.endswith(" ...")
    assert len(marked) < 80


def test_mark_span_rejects_bad_offsets() -> None:
    with pytest.raises(ValueError):
        mark_span(CONTEXT, 5, 5)


def test_locate_falls_back_to_search_when_offsets_disagree() -> None:
    assert locate(CONTEXT, {"text": "Panthers", "start": 0, "end": 3}) == (21, 29)
    assert locate(CONTEXT, {"text": "Ravens", "start": -1, "end": -1}) is None


def test_training_pairs_use_gold_and_wrong_reader_spans() -> None:
    candidates = {
        "q1": [
            {"text": "The Broncos", "start": 0, "end": 11},  # EM-equal to gold: skipped
            {"text": "Panthers", "start": 21, "end": 29},
            {"text": "Super Bowl 50", "start": 43, "end": 56},
            {"text": "24 to 10", "start": 30, "end": 38},
        ]
    }
    pairs = training_pairs([_q()], candidates, negatives_per_question=2)
    assert [p.label for p in pairs] == [1, 0, 0]
    assert f"{START} Broncos {END}" in pairs[0].marked_context
    assert f"{START} Panthers {END}" in pairs[1].marked_context
    assert f"{START} Super Bowl 50 {END}" in pairs[2].marked_context


def test_training_pairs_skip_misaligned_gold_offsets() -> None:
    assert training_pairs([_q(starts=(5,))], {"q1": []}) == []


def test_scoring_inputs_keep_rank_order() -> None:
    candidates = {"q1": [{"text": "Broncos", "start": 4, "end": 11}, {"text": "Ravens"}]}
    rows = scoring_inputs([_q()], candidates, top_k=2)
    assert [(r[0], r[1]) for r in rows] == [("q1", 0), ("q1", 1)]
    assert rows[1][3] == f"{START} Ravens {END}"
