import json
import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# CUDA device
device = "cuda:0"

# Paths / constants
PRETRAINED_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CLIP_MODEL_PATH = "Bingsu/clip-vit-large-patch14-ko"
PROMPT_FILE = "./benchmark/b-kc.json"


def _read_prompt():
    with open(PROMPT_FILE, "rb") as f:
        return json.load(f)


def sanitize_filename(s: str, max_length: int = 50) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in s)
    return safe[:max_length].rstrip("_")


def run_qualitative_evaluation(model_names):
    prompts = _read_prompt()
    for model_name in model_names:
        model_dir = Path(f"../model-weights/{model_name}")
        
        # Look for LoRA weights directly in the model directory
        lora_path = model_dir / "pytorch_lora_weights.safetensors"
        if not lora_path.is_file():
            tqdm.write(f"[WARN] No LoRA weights at {lora_path}, skipping model {model_name}.")
            continue
        # Load pipeline with LoRA weights
        pipeline = DiffusionPipeline.from_pretrained(
            PRETRAINED_MODEL_PATH,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL_PATH)
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)
        pipeline.text_encoder = text_encoder.to(device)
        pipeline.tokenizer = tokenizer
        pipeline.load_lora_weights(str(lora_path))
        pipeline = pipeline.to(device)
        
        test_name = PROMPT_FILE.split("/")[-1].split(".")[0]
        save_dir = Path(f"./generated-results/{test_name}/{model_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for lang, prompt_list in prompts.items():
            lang_dir = save_dir / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            for idx, prompt in enumerate(tqdm(prompt_list, desc=f"{model_name}", unit="img")):
                fname = f"{idx:03d}.png"
                out_path = lang_dir / fname
                if out_path.exists():
                    print(f"[SKIP] {out_path} already exists.")
                    continue

                with torch.autocast(device):
                    image = pipeline(prompt, num_inference_steps=50).images[0]

                image.save(out_path)
                print(f"[SAVED] {model_name} → {out_path}")


if __name__ == "__main__":
    models = ["kodi"]
    run_qualitative_evaluation(models)
    print("Done")
