"""Answer verification and a learned switch for the margin-triggered reranker.

Pure-Python and numpy modules live here so they are tested without a GPU:
SQuAD loading and splitting (``squad``), candidate labelling and features
(``features``), the switch rules and their tuning (``switch``), and the guard
that keeps the dev set to a single final read (``protocol``). Model code is in
``scripts/verifier/``.
"""
