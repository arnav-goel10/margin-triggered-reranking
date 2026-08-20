#!/usr/bin/env python3
from transformers import pipeline
import argparse
import json
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = str(ROOT / "models" / "roberta_base_d2e5_wd01_ep2_acc2")
DEFAULT_DEV = str(ROOT / "data" / "dev-v1.1.json")
DEFAULT_OUT = str(ROOT / "predictions" / "candidates" / "predictions_with_5_acc2.json")


def main():
    parser = argparse.ArgumentParser(description="Run QA inference on SQuAD dev and write top-k predictions")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL, help="Path or HF id of the fine-tuned model")
    parser.add_argument("--dev_file", type=str, default=DEFAULT_DEV, help="Path to SQuAD dev-v1.1.json")
    parser.add_argument("--out_file", type=str, default=DEFAULT_OUT, help="Where to write predictions_with_5.json")
    parser.add_argument("--top_k", type=int, default=5, help="Number of candidates to return per question")
    args = parser.parse_args()

    print("Loading model...")
    device_id = 0 if torch.cuda.is_available() else -1
    qa = pipeline("question-answering", model=args.model_path, tokenizer=args.model_path, device=device_id)
    print(f"Model loaded (device={'cuda:0' if device_id == 0 else 'cpu'})")

    print("Loading SQuAD dev dataset...")
    with open(args.dev_file, "r") as f:
        squad_dict = json.load(f)

    predictions = {}
    count = 0

    for article in squad_dict["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa_item in paragraph["qas"]:
                question = qa_item["question"]
                qid = qa_item["id"]

                # handle different pipeline versions
                try:
                    result = qa({"context": context, "question": question}, top_k=args.top_k)
                except TypeError:
                    result = qa({"context": context, "question": question}, topk=args.top_k)

                if isinstance(result, dict):
                    result = [result]

                candidates = []
                for r in result:
                    text = r.get("answer") or r.get("text") or ""
                    candidates.append({
                        "text": text,
                        "score": float(r.get("score", 0.0)),
                        "start": int(r.get("start", -1)),
                        "end": int(r.get("end", -1)),
                    })

                predictions[qid] = candidates[: args.top_k]
                count += 1

    print(f"Generated top-{args.top_k} predictions for {count} questions")
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved to {args.out_file}")


if __name__ == "__main__":
    main()