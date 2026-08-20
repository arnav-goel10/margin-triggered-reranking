#!/usr/bin/env python3
# Complete workflow script for SQuAD QA with margin-triggered reranking
# This script automates the entire pipeline:
# 1. Generate top-k candidate predictions
# 2. Apply margin-triggered reranking
# 3. Evaluate results
#
# Best default parameters (from hyperparameter search):
# - top_k: 2
# - alpha: 0.5
# - min_gap: 0.05
# - reranker_type: bi_encoder
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Defaults resolved relative to this file
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = str(ROOT / "models" / "roberta_base_d2e5_wd01_ep2_acc2")
DEFAULT_DEV = str(ROOT / "data" / "dev-v1.1.json")
MODEL_DRIVE_ID = "1XNCI0GWPADil13jA2u0uug43mSDTsnA6"
MODEL_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={MODEL_DRIVE_ID}"

# Best parameters from hyperparameter search
DEFAULT_TOP_K = 2
DEFAULT_ALPHA = 0.5
DEFAULT_MIN_GAP = 0.05
DEFAULT_RERANKER_TYPE = "bi_encoder"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def check_and_download_model(model_path):
    # Check if model exists, download from Google Drive if not
    model_dir = Path(model_path)
    
    # Check if model directory exists and has required files
    if model_dir.exists() and (model_dir / "config.json").exists():
        print(f"Model found at: {model_path}")
        return
    
    print(f"\n{'='*60}")
    print("Model not found. Downloading from Google Drive...")
    print(f"{'='*60}")
    
    # Create models directory if it doesn't exist
    models_dir = model_dir.parent
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using gdown (more reliable than direct requests)
    zip_path = models_dir / "roberta_best.zip"
    
    print(f"Downloading model (4.5GB) to {zip_path}...")
    print("This may take several minutes depending on your connection.")
    
    # try using gdown first
    try:
        import gdown  # type: ignore
        gdown.download(MODEL_DRIVE_URL, str(zip_path), quiet=False)
    except ImportError:
        print("\n'gdown' not installed. Installing it now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
        import gdown  # type: ignore
        gdown.download(MODEL_DRIVE_URL, str(zip_path), quiet=False)
    except Exception as e:
        print(f"\nError downloading with gdown: {e}")
        print("\nAlternative: Please download manually from:")
        print(f"   https://drive.google.com/file/d/{MODEL_DRIVE_ID}/view?usp=sharing")
        print(f"   Extract to: {model_dir}")
        sys.exit(1)
    
    # Extract the zip file
    print(f"\nExtracting model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the root directory name from the zip
        zip_members = zip_ref.namelist()
        if zip_members:
            # Extract to a temp location first
            temp_extract = models_dir / "_temp_extract"
            zip_ref.extractall(temp_extract)
            
            # Find the actual model directory in the extracted files
            extracted_dirs = [d for d in temp_extract.iterdir() if d.is_dir()]
            if extracted_dirs:
                # Move the extracted directory to the target location
                extracted_dir = extracted_dirs[0]
                if extracted_dir.name != model_dir.name:
                    # If names don't match, rename it
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
                    extracted_dir.rename(model_dir)
                else:
                    # If it's already in the right place, just move contents
                    if not model_dir.exists():
                        extracted_dir.rename(model_dir)
                    else:
                        for item in extracted_dir.iterdir():
                            shutil.move(str(item), str(model_dir / item.name))
                        extracted_dir.rmdir()
            
            # Clean up temp directory if it still exists
            if temp_extract.exists():
                shutil.rmtree(temp_extract)
    
    # Clean up zip file
    zip_path.unlink()
    
    # Verify extraction
    if model_dir.exists() and (model_dir / "config.json").exists():
        print(f"Model downloaded and extracted successfully!")
        print(f"Location: {model_path}")
    else:
        print(f"Warning: Model extraction may have failed. Please verify: {model_path}")


def run_command(cmd, description):
    # Run a command and handle errors
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"{description} completed successfully")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Complete workflow: generate candidates → rerank → evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Best default parameters (from hyperparameter search):
  top_k: {DEFAULT_TOP_K}
  alpha: {DEFAULT_ALPHA}
  min_gap: {DEFAULT_MIN_GAP}
  reranker_type: {DEFAULT_RERANKER_TYPE}

Example usage:
  # Use best defaults
  python src/run_complete_workflow.py

  # Custom parameters
  python src/run_complete_workflow.py --top_k 3 --alpha 0.6 --min_gap 0.1

  # Skip candidate generation (use existing file)
  python src/run_complete_workflow.py --skip_candidates --nbest_file predictions/candidates/predictions_with_2_acc2.json
        """
    )
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL,
                        help=f"Path to fine-tuned model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dev_file", type=str, default=DEFAULT_DEV,
                        help=f"Path to SQuAD dev-v1.1.json (default: {DEFAULT_DEV})")
    
    # Candidate generation
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of candidates to generate (default: {DEFAULT_TOP_K})")
    parser.add_argument("--skip_candidates", action="store_true",
                        help="Skip candidate generation step (use existing nbest_file)")
    parser.add_argument("--nbest_file", type=str, default=None,
                        help="Path to existing nbest file (auto-generated if not provided)")
    
    # Reranking parameters
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help=f"Fusion weight for baseline score (default: {DEFAULT_ALPHA})")
    parser.add_argument("--min_gap", type=float, default=DEFAULT_MIN_GAP,
                        help=f"Margin threshold - only rerank if score1 - score2 < min_gap (default: {DEFAULT_MIN_GAP})")
    parser.add_argument("--reranker_type", type=str, default=DEFAULT_RERANKER_TYPE,
                        choices=["bi_encoder", "cross_encoder"],
                        help=f"Reranker type (default: {DEFAULT_RERANKER_TYPE})")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Bi-encoder model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--cross_model_name", type=str, default="cross-encoder/nli-deberta-base",
                        help="Cross-encoder model name (only used if reranker_type=cross_encoder)")
    
    # Output paths
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Prefix for output files (auto-generated from parameters if not provided)")
    parser.add_argument("--candidates_dir", type=str, default="predictions/candidates",
                        help="Directory for candidate files (default: predictions/candidates)")
    parser.add_argument("--reranked_dir", type=str, default="predictions/reranked",
                        help="Directory for reranked files (default: predictions/reranked)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for result files (default: results)")
    
    args = parser.parse_args()
    
    # Generate output file names
    if args.output_prefix is None:
        alpha_str = f"{int(args.alpha * 100):02d}"
        gap_str = f"{int(args.min_gap * 1000):03d}"
        args.output_prefix = f"top{args.top_k}_alpha{alpha_str}_gap{gap_str}_{args.reranker_type}"
    
    candidates_file = args.nbest_file
    if candidates_file is None:
        candidates_file = f"{args.candidates_dir}/predictions_with_{args.top_k}_acc2.json"
    
    reranked_file = f"{args.reranked_dir}/predictions_{args.output_prefix}.json"
    results_file = f"{args.results_dir}/results_{args.output_prefix}.json"
    
    # Ensure directories exist
    for dir_path in [args.candidates_dir, args.reranked_dir, args.results_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting Complete Workflow")
    print("="*60)
    
    # Check and download model if needed
    check_and_download_model(args.model_path)
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Dev file: {args.dev_file}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Min gap: {args.min_gap}")
    print(f"  Reranker type: {args.reranker_type}")
    print(f"\nOutput files:")
    print(f"  Candidates: {candidates_file}")
    print(f"  Reranked: {reranked_file}")
    print(f"  Results: {results_file}")
    
    # Step 1: Generate candidates (unless skipped)
    if not args.skip_candidates:
        cmd = [
            sys.executable, "src/evaluate_k_candidates.py",
            "--model_path", args.model_path,
            "--dev_file", args.dev_file,
            "--out_file", candidates_file,
            "--top_k", str(args.top_k),
        ]
        run_command(cmd, "Step 1: Generate top-k candidates")
    else:
        print(f"\nSkipping candidate generation (using existing file: {candidates_file})")
        if not Path(candidates_file).exists():
            print(f"Error: Candidate file not found: {candidates_file}")
            sys.exit(1)
    
    # Step 2: Apply margin-triggered reranking
    cmd = [
        sys.executable, "src/rerank_squad_candidates_threshold.py",
        "--dev_file", args.dev_file,
        "--nbest_file", candidates_file,
        "--out_file", reranked_file,
        "--alpha", str(args.alpha),
        "--min_gap", str(args.min_gap),
        "--normalize", "minmax",
        "--candidate_text_mode", "answer_is",
        "--cap_topk", str(args.top_k),
        "--reranker_type", args.reranker_type,
        "--batch_size", "32",
    ]
    
    if args.reranker_type == "bi_encoder":
        cmd.extend(["--model_name", args.model_name])
    else:
        cmd.extend(["--cross_model_name", args.cross_model_name])
    
    run_command(cmd, "Step 2: Apply margin-triggered reranking")
    
    # Step 3: Evaluate results
    cmd = [
        sys.executable, "src/evaluate-v2.0.py",
        args.dev_file,
        reranked_file,
        "--out-file", results_file,
    ]
    run_command(cmd, "Step 3: Evaluate results")
    
    # Print summary
    print("\n" + "="*60)
    print("Workflow Complete!")
    print("="*60)
    print(f"\nResults saved to: {results_file}")
    
    # try to read and show results
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"\nFinal Metrics:")
        print(f"  EM: {results.get('exact', 0):.2f}%")
        print(f"  F1: {results.get('f1', 0):.2f}%")
        print(f"  Total: {results.get('total', 0)}")
    except Exception as e:
        print(f"\nCouldn't read results: {e}")
    
    print(f"\nAll output files:")
    print(f"  Candidates: {candidates_file}")
    print(f"  Reranked: {reranked_file}")
    print(f"  Results: {results_file}")
    print()


if __name__ == "__main__":
    main()

