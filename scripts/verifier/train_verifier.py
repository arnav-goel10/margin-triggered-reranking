#!/usr/bin/env python3
"""Fine-tune a cross-encoder to tell the right answer span from the reader's wrong ones.

Positives are gold spans of reader_train questions; negatives are the reader's
highest-ranked wrong candidates for the same questions. reader_val pairs pick
the checkpoint. Neither the held-out slice nor dev is read here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verifier import squad  # noqa: E402
from verifier.pairs import END, START, training_pairs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="google/electra-base-discriminator")
    parser.add_argument("--train-squad", type=Path, required=True)
    parser.add_argument("--train-candidates", type=Path, required=True)
    parser.add_argument("--val-squad", type=Path, required=True)
    parser.add_argument("--val-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; refusing to run on CPU")

    def pairs_for(squad_file: Path, cand_file: Path):
        questions = list(squad.iter_questions(squad.load(squad_file)))
        candidates = json.loads(cand_file.read_text())
        return training_pairs(questions, candidates, negatives_per_question=args.negatives)

    train_pairs = pairs_for(args.train_squad, args.train_candidates)
    val_pairs = pairs_for(args.val_squad, args.val_candidates)
    for name, pairs in (("train", train_pairs), ("val", val_pairs)):
        pos = sum(p.label for p in pairs)
        print(f"{name}: {len(pairs)} pairs, {pos} positive, {len(pairs) - pos} negative")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": [START, END]})
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    def encode(pairs):
        ds = Dataset.from_dict(
            {
                "question": [p.question for p in pairs],
                "context": [p.marked_context for p in pairs],
                "label": [p.label for p in pairs],
            }
        )
        return ds.map(
            lambda b: tokenizer(
                b["question"], b["context"], truncation="only_second", max_length=args.max_length
            ),
            batched=True,
            remove_columns=["question", "context"],
        )

    train_ds, val_ds = encode(train_pairs), encode(val_pairs)

    def metrics(pred):
        logits, labels = pred
        margin = logits[:, 1] - logits[:, 0]
        accuracy = float(((margin > 0).astype(int) == labels).mean())
        return {"accuracy": accuracy, "mean_margin_pos": float(margin[labels == 1].mean())}

    bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    training_args = TrainingArguments(
        output_dir=str(args.out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        bf16=bf16,
        fp16=not bf16,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=200,
        report_to=[],
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics,
    )
    trainer.train()
    final = args.out_dir / "model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    summary = {
        "base_model": args.base_model,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "negatives_per_question": args.negatives,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "val_metrics": trainer.evaluate(),
        "log_history": trainer.state.log_history,
    }
    (args.out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: np.asarray(o).tolist()) + "\n"
    )
    print(f"saved verifier to {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
