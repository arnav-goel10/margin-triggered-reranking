#!/usr/bin/env python3
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
import argparse
import json
import torch
from pathlib import Path


def load_and_flatten_squad(path: str) -> Dataset:
    # load squad data and flatten it
    with open(path, "r") as f:
        squad_dict = json.load(f)

    contexts, questions, answers = [], [], []
    for article in squad_dict["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if len(qa["answers"]) > 0:
                    ans = qa["answers"][0]
                    answers.append({"text": ans["text"], "answer_start": ans["answer_start"]})
                else:
                    answers.append({"text": "", "answer_start": 0})
                contexts.append(context)
                questions.append(question)
    return Dataset.from_dict({"context": contexts, "question": questions, "answers": answers})


def build_tokenizer_and_model(model_name: str):
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model


def preprocess_function(examples, tokenizer):
    # preprocess examples for training
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = inputs.sequence_ids(i)
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]

        if answer["text"] == "" or answer["answer_start"] == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        token_start_index = 0
        while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(input_ids) - 1
        while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)
            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a QA model on SQuAD v1.1")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="HF model id (e.g., roberta-base, distilroberta-base)")
    parser.add_argument("--train_path", type=str, default="../data/train-v1.1.json")
    parser.add_argument("--dev_path", type=str, default="../data/dev-v1.1.json")
    parser.add_argument("--output_dir", type=str, default="../models/roberta_finetuned")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_batch", type=int, default=8)
    parser.add_argument("--eval_batch", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--max_train", type=int, default=2000,
                        help="Max training examples (set -1 for full)")
    parser.add_argument("--max_eval", type=int, default=500,
                        help="Max eval examples (set -1 for full)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_path = str((root / args.train_path).resolve()) if args.train_path.startswith("..") else args.train_path
    dev_path = str((root / args.dev_path).resolve()) if args.dev_path.startswith("..") else args.dev_path
    output_dir = str((root / args.output_dir).resolve()) if args.output_dir.startswith("..") else args.output_dir

    print("Loading SQuAD dataset...")
    train_ds = load_and_flatten_squad(train_path)
    eval_ds = load_and_flatten_squad(dev_path)
    dataset = DatasetDict({"train": train_ds, "validation": eval_ds})
    print(f"Train: {len(dataset['train'])}, Eval: {len(dataset['validation'])}")

    tokenizer, model = build_tokenizer_and_model(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    print(f"Using device: {device}")

    print("Tokenizing...")
    tokenized = dataset.map(lambda ex: preprocess_function(ex, tokenizer),
                            batched=True,
                            remove_columns=dataset["train"].column_names)

    train_tokenized = tokenized["train"] if args.max_train == -1 else tokenized["train"].select(range(min(args.max_train, len(tokenized["train"]))))
    eval_tokenized = tokenized["validation"] if args.max_eval == -1 else tokenized["validation"].select(range(min(args.max_eval, len(tokenized["validation"]))))

    # check what precision to use
    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not supports_bf16

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        save_total_limit=3,
        bf16=supports_bf16,
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Done. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()