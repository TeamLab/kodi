# KODI: A Korean Diffusion Model for Bilingual Text-to-Image Generation and Cultural Fidelity

KODI is a diffusion model that generates high-quality Korean cultural images from Korean text prompts. It uses a Korean CLIP-based text encoder to better understand Korean prompts and generate culturally appropriate images.


## 📦 Installation

```bash
git clone https://github.com/TeamLab/kodi.git
cd kodi
pip install -r requirements.txt
```

## 📊 Datasets and Benchmarks

### Korean Cultural Dataset (KCD)
Our Korean cultural training dataset is located at:
```
korean-cultural-dataset/
```
This dataset contains Korean cultural images with corresponding Korean text descriptions.

### MC-K Evaluation Benchmark
The Korean cultural evaluation benchmark MC-K is available at:
```
evaluation/benchmark/
```
This benchmark is used for evaluating cultural appropriateness and Korean language understanding.



## 🔧 Usage

### 1. Model Training

```bash
# Train KODI with Korean Cultural Dataset (KCD)
python training/train_kodi.py --config training/configs/kodi.py
```

### 2. Image Generation

```bash
# Generate images with evaluation dataset
python evaluation/generate_eval_images.py
```

### 3. Model Evaluation

```bash
# Evaluate with MC-K benchmark using KC-CLIP
python evaluation/evaluate_by_kcclip.py
```

## 🤗 Model Weights

| Model | Type | Location | Description |
|-------|------|----------|-------------|
| **KODI** | LoRA Weights | `model-weights/kodi/` | Korean cultural diffusion model (included in repository) |
| **KODI Base** | Foundation Model | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | Stable Diffusion v1.5 base model |
| **KC-CLIP KO** | Evaluator | [letgoofthepizza/kc-clip-ko](https://huggingface.co/letgoofthepizza/kc-clip-ko) | Korean cultural CLIP model (Korean) |
| **KC-CLIP EN** | Evaluator | [letgoofthepizza/kc-clip-en](https://huggingface.co/letgoofthepizza/kc-clip-en) | Korean cultural CLIP model (English) |
