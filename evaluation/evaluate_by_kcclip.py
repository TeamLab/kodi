#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Default paths (can be overridden by command line arguments)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_WEIGHTS_DIR = BASE_DIR.parent / "model-weights" / "evaluator"
DEFAULT_PROMPT_FILE = BASE_DIR / "benchmark/mc-k.json"
DEFAULT_GEN_ROOT = BASE_DIR / "generated-results/mc-k"
DEFAULT_OUT_ROOT = BASE_DIR / "evaluation-results"

CATEGORY_BOUNDS = {
    "architecture": (0, 186),
    "landmark": (186, 247),
    "food": (247, 396),
    "clothes": (396, 520),
}


def idx_to_category(idx: int) -> str:
    for cat, (start, end) in CATEGORY_BOUNDS.items():
        if start <= idx < end:
            return cat
    return "unknown"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_load(data_sort, model_weights_dir):
    """Load CLIP model based on language"""
    # ko -> ko, en_rom/en_sem -> en
    lang = "ko" if data_sort == "ko" else "en"
    model_path = model_weights_dir / f"kc-clip-{lang}"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = CLIPModel.from_pretrained(str(model_path), local_files_only=True).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(str(model_path), local_files_only=True)
    return model, processor


def compute_score(img: Image.Image, text: str, model, processor) -> float:
    inputs = processor(text=[text], images=img, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)

        img_e = out.image_embeds  # (1, D)
        txt_e = out.text_embeds  # (1, D)
        return F.cosine_similarity(img_e, txt_e, dim=-1).item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP scores for generated images")
    parser.add_argument("--prompt_file", type=str, default=str(DEFAULT_PROMPT_FILE),
                       help="Path to JSON file containing prompts")
    parser.add_argument("--gen_root", type=str, default=str(DEFAULT_GEN_ROOT),
                       help="Root directory containing generated images")
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT),
                       help="Output directory for evaluation results")
    parser.add_argument("--model_weights_dir", type=str, default=str(DEFAULT_MODEL_WEIGHTS_DIR),
                       help="Directory containing CLIP model weights")
    parser.add_argument("--target_models", nargs="+", default=["kodi"],
                       help="List of target model names to evaluate")
    
    args = parser.parse_args()
    
    prompt_file = Path(args.prompt_file)
    gen_root = Path(args.gen_root)
    out_root = Path(args.out_root)
    model_weights_dir = Path(args.model_weights_dir)
    target_models = set(args.target_models)
    
    # Check if required files/directories exist
    if not prompt_file.exists():
        print(f"Error: Prompt file not found at {prompt_file}")
        return
    
    if not gen_root.exists():
        print(f"Error: Generated images directory not found at {gen_root}")
        return
    
    if not model_weights_dir.exists():
        print(f"Error: Model weights directory not found at {model_weights_dir}")
        return
    
    prompts = read_json(prompt_file)

    # Get list of model directories
    model_dirs = [d for d in gen_root.iterdir() if d.is_dir() and d.name in target_models]
    
    if not model_dirs:
        print(f"Warning: No target models found in {gen_root}")
        print(f"Available models: {[d.name for d in gen_root.iterdir() if d.is_dir()]}")
        return

    for model_dir in sorted(model_dirs, key=lambda x: x.name):
        model_name = model_dir.name
        print(f"▶ Evaluating model: {model_name}")

        model_out_root = out_root / model_name
        model_out_root.mkdir(parents=True, exist_ok=True)

        all_lang_scores = {}

        for lang_dir in sorted(model_dir.iterdir(), key=lambda p: p.name):
            if not lang_dir.is_dir():
                continue
                
            lang = lang_dir.name
            
            # Check if language is supported
            try:
                model, processor = _model_load(lang, model_weights_dir)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

            lang_results = {
                "overall": {"avg_score": 0.0},
                "by_category": {cat: {"scores": {}, "avg_score": 0.0} for cat in CATEGORY_BOUNDS},
            }

            overall_sum = 0.0
            overall_cnt = 0
            cat_sums = {cat: 0.0 for cat in CATEGORY_BOUNDS}
            cat_cnts = {cat: 0 for cat in CATEGORY_BOUNDS}

            for img_file in tqdm(sorted(lang_dir.iterdir()), desc=f"{model_name}@{lang}", unit="img"):
                if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                idx = int(img_file.stem)
                if idx >= len(prompts.get(lang, [])):
                    continue

                prompt = prompts[lang][idx]
                img = Image.open(img_file).convert("RGB")
                score = compute_score(img, prompt, model, processor)

                overall_sum += score
                overall_cnt += 1

                cat = idx_to_category(idx)
                if cat in CATEGORY_BOUNDS:
                    lang_results["by_category"][cat]["scores"][f"{idx:03d}"] = {
                        "prompt": prompt,
                        "score": round(score, 5),
                    }
                    cat_sums[cat] += score
                    cat_cnts[cat] += 1

            if overall_cnt:
                lang_results["overall"]["avg_score"] = round(overall_sum / overall_cnt, 5)
            for cat in CATEGORY_BOUNDS:
                if cat_cnts[cat]:
                    lang_results["by_category"][cat]["avg_score"] = round(cat_sums[cat] / cat_cnts[cat], 5)

            all_lang_scores[lang] = lang_results

        out_file = model_out_root / f"{model_name}_clip_scores.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_lang_scores, f, ensure_ascii=False, indent=2)

        for lang, res in all_lang_scores.items():
            print(f"[{model_name}][{lang}] overall_avg={res['overall']['avg_score']}", end="")
            for cat, cdata in res["by_category"].items():
                print(f" | {cat}_avg={cdata['avg_score']}", end="")
            print()


if __name__ == "__main__":
    main()
