# Efficient-HA 🚀

Benchmarking and experimentation environment for **vision-language models (VLMs)** such as **Gemma-3, Qwen-VL, Paligemma**, and related architectures.

This repo contains scripts, environment files, and experiment logs for reproducible evaluations on **COCO2014**, **Flickr8k**, and other multimodal datasets.

---

## 📦 Environment Setup

We recommend using **mamba** (faster conda) for dependency management.

### Create environment
```bash
mamba env create -f environment.yml
mamba activate EHA-mayflower
uv pip install -r requirements.txt
```

```**bash**
git clone https://github.com/mayflower6157/Efficient-HA.git
cd transformers
pip install -e .
```

## Usage: Running Benchmark
```bash
python generate_response_chair.py \
  --model google/gemma-3-4b-it \
  --data_path /mnt/disks/extra-disk/datasets/coco2014 \
  --device cuda \
  --dtype bfloat16
```
