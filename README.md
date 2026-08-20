# Margin-Triggered Answer Reranking

Extractive question answering on SQuAD v1.1. A fine-tuned RoBERTa-base reader picks an
answer span, and a second-stage reranker steps in **only when the reader is unsure**.
The trigger is the confidence margin between the reader's top two candidate spans: when
that gap is wide the reader is left alone, and when it is narrow a bi-encoder rescores
the two candidates against the question.

The point of the design is restraint. A reranker that touches every question does more
damage than good on a strong baseline, because it overturns answers the reader already
had right. Gating on the margin keeps the intervention to **55 of 10,570 questions
(0.5%)**, and of those 55 it fixes 14 and breaks 1.

## Results

SQuAD v1.1 dev set, 10,570 questions.

| System | EM | F1 | Predictions changed |
| --- | --- | --- | --- |
| RoBERTa-base reader (baseline) | 84.276 | 90.926 | n/a |
| **+ margin-triggered reranking** | **84.399** | **91.038** | **55 (0.5%)** |

Breakdown of the 55 changed predictions:

| Outcome | Count |
| --- | --- |
| Wrong then right | 14 |
| Right then wrong | 1 |
| Wrong then still wrong | 40 |
| Right then still right | 0 |

Net gain is 13 questions, which is the whole of the +0.123 EM. The 14-to-1 ratio matters
more than the headline number: it says the gate is firing on the right questions.

## Why gate on the margin

Two measurements from the baseline reader motivate the whole approach. Both are
reproducible with the scripts in `src/`.

**Where the correct answer actually sits.** Counting how often the gold answer appears at
each rank among the reader's top 5 candidates (`count_gold_candidate_positions.py`):

| Rank of gold answer | Questions | Share |
| --- | --- | --- |
| 1 | 8,908 | 84.3% |
| 2 | 661 | 6.3% |
| 3 | 253 | 2.4% |
| 4 | 150 | 1.4% |
| 5 | 81 | 0.8% |
| Not in top 5 | 517 | 4.9% |

Reranking the top two spans has a hard ceiling of 90.5% EM, and 6.3 points of that is
reachable. Everything below rank 2 is out of scope by construction, which is why the
system only ever looks at two candidates.

**The margin separates those cases.** Score gap between candidate 1 and candidate 2
(`calculate_candidate_stats.py`):

| | Median gap | 25th percentile |
| --- | --- | --- |
| Gold answer is candidate 1 | 0.602 | 0.233 |
| Gold answer is candidate 2 | 0.113 | 0.038 |

When the reader is right, it is usually right by a wide margin. When the correct answer is
actually its second choice, the two scores sit close together. That separation is the
signal the gate keys on.

## How it works

1. The fine-tuned RoBERTa reader produces the top-k candidate spans per question with scores.
2. Compute the margin `score(c1) - score(c2)`.
3. If the margin is **above** `min_gap`, keep candidate 1 and stop.
4. If the margin is **at or below** `min_gap`, rescore both candidates:

   ```
   final(c) = alpha * reader_score_normalised(c) + (1 - alpha) * similarity(question, c)
   ```

   `alpha` is the weight on the reader, so `alpha=0.5` trusts both halves equally.
   Reader scores are min-max normalised per question before blending, and `similarity`
   is cosine similarity from `sentence-transformers/all-MiniLM-L6-v2`.
5. Return whichever candidate has the higher blended score.

Best configuration from the search: `top_k=2`, `alpha=0.5`, `min_gap=0.05`,
bi-encoder reranker.

## Quickstart

Requires Python 3.8 or newer.

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
  --epochs 2 --gradient_accumulation 2 --weight_decay 0.01 \
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

# Margin-triggered reranking, best configuration
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

Key files: `results/best_results_top2_alpha05_gap015_bienc.json` (final),
`results/results_acc2_best_original.json` (baseline),
`results/hparam_search_top2_3_5_biencoder.csv` (all 60 runs).

## Limitations

- **The gain is small.** +0.123 EM on a strong baseline. The honest reading is that the
  gate successfully avoids harm, not that reranking transforms the system.
- **`alpha` dominates `min_gap`.** At `alpha=0.5` every threshold from 0.05 to 0.25
  produces identical output, so the gap stops binding once the blend weight is balanced.
  The threshold matters at lower alpha, where the search shows it clearly.
- **Top-2 only.** 4.9% of questions have no correct span in the top 5 at all. No amount of
  reranking reaches those.
- **Bi-encoder over cross-encoder.** The cross-encoder was tested (`results/experimental/`)
  and scored worse while costing more per query.

## License

MIT, see [LICENSE](LICENSE). The SQuAD dataset is CC BY-SA 4.0.
`src/evaluate-v2.0.py` is the official Stanford evaluation script, included unmodified.
