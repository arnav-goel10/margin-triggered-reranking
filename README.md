# Margin-Triggered Answer Reranking

Extractive question answering on SQuAD v1.1. A fine-tuned RoBERTa-base reader picks an
answer span. The question this project asks is whether a second-stage reranker can fix the
reader's mistakes if it only steps in when the reader is unsure, measured by the confidence
gap between its top two candidates.

**Short answer: the uncertainty signal is real, but off-the-shelf rerankers can't use it.**
When the reader is wrong and the right answer is its second choice, the gap between its top
two scores is about 5 times smaller than when it is right. But when a MiniLM bi-encoder or
cross-encoder overrides the reader inside that narrow band, it breaks more answers than it
fixes, even at the tightest threshold.

## Results

SQuAD v1.1 dev set, 10,570 questions. Reader baseline: **84.276 EM, 90.926 F1**.

Overriding the reader with the reranker whenever they disagree and the top-two gap is at or
below the threshold (top 2 candidates, `alpha` below 0.5):

| Gap threshold | Bi-encoder: fixed / broken | EM | Cross-encoder: fixed / broken | EM |
| --- | --- | --- | --- | --- |
| 0.05 | 107 / 136 | 84.002 | 112 / 133 | 84.078 |
| 0.10 | 158 / 248 | 83.425 | 163 / 239 | 83.557 |
| 0.15 | 200 / 344 | 82.914 | 204 / 322 | 83.160 |
| 0.20 | 225 / 426 | 82.375 | 226 / 397 | 82.658 |
| 0.25 | 240 / 511 | 81.712 | 238 / 469 | 82.091 |

Every row lowers exact match, and the wider the band the worse it gets. Even in the
narrowest band, where the reader is least sure, the reader is right more often than the
reranker that disagrees with it. Full grid in `results/hparam_search_top2_3_5_*.csv`.

### A correction to an earlier version of this README

An earlier version reported **84.399 EM (+0.123)** at `alpha=0.5`, changing 55 predictions
(14 fixed, 1 broken). That result does not hold up, for three reasons:

1. Reader and reranker scores are min-max normalised per question. With two candidates
   that maps each pair to (1, 0) or (0, 1), so at `alpha=0.5` a disagreement blends to an
   exact tie. The 55 changes come from floating-point tie-breaks. This is
   also why every gap threshold gave identical output at `alpha=0.5`.
2. `alpha` therefore has only two real settings. Below 0.5 the reranker wins every
   disagreement (the table above). Above 0.5 it never changes anything.
3. The configuration was chosen on the same dev set it was reported on.

## Why the margin is still interesting

Two measurements from the baseline reader hold regardless of the reranker. Both are
reproducible with the scripts in `src/`.

**Where the correct answer sits.** How often the gold answer appears at each rank
among the reader's top 5 candidates (`count_gold_candidate_positions.py`):

| Rank of gold answer | Questions | Share |
| --- | --- | --- |
| 1 | 8,908 | 84.3% |
| 2 | 661 | 6.3% |
| 3 | 253 | 2.4% |
| 4 | 150 | 1.4% |
| 5 | 81 | 0.8% |
| Not in top 5 | 517 | 4.9% |

Picking perfectly between the top two would reach 90.5% EM, so there are 6.3 points of
headroom.

**The margin separates those cases.** Score gap between candidate 1 and candidate 2
(`calculate_candidate_stats.py`):

| | Median gap | 25th percentile |
| --- | --- | --- |
| Gold answer is candidate 1 | 0.602 | 0.233 |
| Gold answer is candidate 2 | 0.113 | 0.038 |

The reader knows when it is unsure. The failure is in the second opinion. A general-purpose
sentence-similarity model scores how related a span is to the question, not whether it
answers it, so it can't beat a reader fine-tuned on SQuAD.

## What would fix it

- A second opinion trained for this task, such as a cross-encoder fine-tuned on SQuAD
  train to verify answers, or a second QA model.
- No min-max normalisation with two candidates. Blend raw probabilities, or learn the
  switch rule directly as a classifier on (gap, reranker preference).
- Tune on a held-out slice of SQuAD train and report once on dev.

## Trained verifier experiment (queued, no results yet)

`verifier/`, `scripts/verifier/` and `slurm/` implement those three points in five steps:

1. SQuAD train is split by article into `reader_train` (85%), `reader_val` (5%) and
   `heldout` (10%, 10,477 questions), checked disjoint from each other and from dev
   (`results/verifier/split_summary.json`).
2. A RoBERTa-base reader is retrained on `reader_train` with the published reader's
   settings, so its candidates on `heldout` and dev come from questions it never saw.
3. An ELECTRA-base cross-encoder is fine-tuned on `reader_train` to score a marked
   span against the question: the gold span against the reader's top two wrong
   spans for the same question. Those negatives come from questions the reader
   was trained on, so they are easier than the ones it produces on unseen text;
   `reader_val` measures the verifier on unseen questions.
4. Three switch rules are tuned on `heldout` with 5-fold cross-validation: verifier
   preference alone, the margin gate plus verifier preference, and a logistic
   switch on the gap, reader probabilities and verifier logits. The best
   cross-validated rule is fixed before dev is opened.
5. `final_eval.py --final` reads dev once, reports every rule with fixed and broken
   counts and a paired bootstrap interval on the EM change, and refuses a second read.

Run with `bash slurm/submit.sh` on the cluster. The results table above changes
only when `results/verifier/dev_report.json` exists.

## How it works

1. The fine-tuned RoBERTa reader produces the top-k candidate spans per question with scores.
2. Compute the margin `score(c1) - score(c2)`.
3. If the margin is above `min_gap`, keep candidate 1.
4. Otherwise rescore both candidates with
   `final(c) = alpha * reader_norm(c) + (1 - alpha) * similarity(question, c)`,
   where scores are min-max normalised per question and `similarity` comes from
   `sentence-transformers/all-MiniLM-L6-v2` (bi-encoder) or
   `cross-encoder/ms-marco-MiniLM-L-6-v2`.
5. Return the candidate with the higher blended score.

## Quickstart

Requires Python 3.10 or newer.

```bash
git clone https://github.com/arnav-goel10/margin-triggered-reranking.git
cd margin-triggered-reranking
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh      # SQuAD v1.1 train + dev
python src/run_complete_workflow.py
```

`run_complete_workflow.py` downloads the fine-tuned reader (3.5GB) on first run, generates
candidates, applies the gate, and evaluates. To train the reader yourself instead:

```bash
python src/fine_tune_roberta.py \
  --train_path data/train-v1.1.json \
  --dev_path data/dev-v1.1.json \
  --epochs 2 --gradient_accumulation 2 --weight_decay 0.01 --max_train -1 \
  --output_dir models/roberta_base_d2e5_wd01_ep2_acc2
```

## Repository map

```
src/
  fine_tune_roberta.py               Fine-tune RoBERTa-base on SQuAD v1.1
  evaluate.py                        Baseline top-1 predictions
  evaluate_k_candidates.py           Top-k candidates with scores
  rerank_squad_candidates.py         Reranking (bi-encoder and cross-encoder)
  rerank_squad_candidates_threshold.py   The margin gate
  search_rerank_hyperparams.py       Grid over top_k, alpha, min_gap
  run_complete_workflow.py           End-to-end pipeline

  count_gold_candidate_positions.py  Where the gold answer ranks
  calculate_candidate_stats.py       Score-gap distributions
  evaluate_k_position_stats.py       Accuracy by candidate position
  compare_rerank_stats.py            Which predictions changed, and how
  evaluate-v2.0.py                   Official SQuAD scorer (Stanford, unmodified)

verifier/      Split, verifier inputs, switch rules, protocol guards (tested)
scripts/verifier/  Reader candidates, verifier training and scoring, tuning, dev read
slurm/         Cluster jobs for the verifier experiment
tests/         pytest suite for verifier/ and the CPU pipeline stages
results/       Metrics and the full hyperparameter search
predictions/   Baseline predictions on dev
data/          SQuAD v1.1 (downloaded, not committed)
```

## Reproducing the numbers in this README

```bash
CAND=predictions/candidates/predictions_with_5_acc2.json

# Reader candidates
python src/evaluate_k_candidates.py \
  --dev_file data/dev-v1.1.json --out_file $CAND --top_k 5

# Reranking at one setting (alpha 0.5 ties, see the correction above)
python src/rerank_squad_candidates_threshold.py \
  --dev_file data/dev-v1.1.json --nbest_file $CAND \
  --out_file predictions/predictions_reranked.json \
  --reranker_type bi_encoder --alpha 0.5 --min_gap 0.05 --cap_topk 2

# Score it
python src/evaluate-v2.0.py data/dev-v1.1.json predictions/predictions_reranked.json

# Rank distribution and margin statistics
python src/count_gold_candidate_positions.py \
  --dev_file data/dev-v1.1.json --nbest_file $CAND \
  --max_k 5 --out_file results/gold_position_counts.json

python src/calculate_candidate_stats.py \
  --dev_file data/dev-v1.1.json --nbest_file $CAND \
  --out_file results/candidate_score_stats.json

# Which predictions changed, and in which direction
python src/compare_rerank_stats.py \
  --dev_file data/dev-v1.1.json --nbest_file $CAND \
  --reranked_file predictions/predictions_reranked.json

# Full grid, 60 configurations
python src/search_rerank_hyperparams.py \
  --dev_file data/dev-v1.1.json --nbest $CAND \
  --reranker_type bi_encoder --results_csv results/hparam_search.csv
```

Key files: `results/best_results_top2_alpha05_gap015_bienc.json` (the retracted alpha=0.5 run, kept for the correction above),
`results/results_acc2_best_original.json` (baseline),
`results/hparam_search_top2_3_5_biencoder.csv` (all 60 runs).

## Limitations

- Off-the-shelf rerankers only. Neither model was trained for answer verification.
- Top 2 only. 4.9% of questions have no correct span in the top 5 at all.
- Dev-set only. There is no separate test split, so treat every number here as exploratory.

## License

MIT, see [LICENSE](LICENSE). Third-party terms (the SQuAD dataset, CC BY-SA 4.0, and
the unmodified official Stanford scorer in `src/evaluate-v2.0.py`) are in [NOTICE](NOTICE).
