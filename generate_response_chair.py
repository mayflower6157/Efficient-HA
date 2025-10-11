#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced COCO Caption Evaluation Runner
---------------------------------------
Author: Leo (Ke Xu)
Purpose: Unified DECO/VCD/Greedy/Beam inference pipeline with fault-tolerant logging.
"""

import os, gc, json, time, traceback, argparse, re, sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from loguru import logger

# === Utilities ===
from utils.logger_utils import (
    print_run_chair_header,
    print_run_chair_summary,
    setup_logger,
)
from utils.vcd_add_noise import add_diffusion_noise
from utils.deco_greedy import evolve_deco_greedy, get_early_exit_layers
from utils.vcd_sample import evolve_vcd_sampling


# ============================================
# 🧩 1. Model Loader
# ============================================
def load_model(model_id: str, device: str):
    """Load model + processor with preset resolution caps."""
    min_pixels, max_pixels = 256 * 28 * 28, 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval()
    logger.success(f"Model {model_id} loaded on {device}")
    return model, processor


# ============================================
# 🧩 2. Generation Dispatch
# ============================================
def generate_response(model, processor, args, image_path, question: str):
    """Generate caption response using the specified method."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        method = args.method.lower()
        if method == "greedy":
            gen_kwargs = dict(max_new_tokens=args.max_tokens, do_sample=False)
        elif method == "beam":
            gen_kwargs = dict(max_new_tokens=args.max_tokens, num_beams=5)
        elif method == "dola":
            args.early_exit_layers = get_early_exit_layers(
                model, args.early_exit_layers
            )
            gen_kwargs = dict(
                max_new_tokens=args.max_tokens,
                custom_generate="transformers-community/dola",
                dola_layers=args.early_exit_layers,
                repetition_penalty=1.2,
            )
        elif method == "deco":
            evolve_deco_greedy(model, args)
            gen_kwargs = dict(
                max_new_tokens=args.max_tokens,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        elif method == "vcd":
            evolve_vcd_sampling()
            inputs_cd = inputs.copy()
            inputs_cd["pixel_values"] = add_diffusion_noise(
                inputs["pixel_values"], args.noise_step
            )
            gen_kwargs = dict(
                max_new_tokens=args.max_tokens,
                do_sample=True,
                pixel_values_cd=inputs_cd["pixel_values"].unsqueeze(0).half().cuda(),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
            )
        else:
            raise ValueError(f"❌ Unknown generation method: {args.method}")

        outputs = model.generate(**inputs, **gen_kwargs)
        if hasattr(outputs, "sequences"):
            outputs = outputs.sequences

        # Decode output text
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, outputs)]
        text_out = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return text_out.strip()


# ============================================
# 🧩 3. JSON Processing Loop
# ============================================
def process_json(model, processor, args, output_json, save_every=20):
    input_jsonl = Path("/home/mayflower/Efficient-HA/opera_log/llava-1.5/greedy.jsonl")
    with open(input_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
        for s in samples:
            s.pop("caption", None)
    total = len(samples)
    logger.info(f"Loaded {total} samples.")

    processed = []
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            processed = json.load(f)
    done_ids = {item["image_id"] for item in processed}
    buffer, errors = [], []
    logger.info(f"Resuming from {len(done_ids)} completed samples.")

    q = "Describe this image in detail."
    start = time.time()

    for i, entry in enumerate(
        tqdm(samples, total=total, desc="Processing", unit="img")
    ):
        img_id = entry["image_id"]
        if img_id in done_ids:
            continue

        image_path = os.path.join(args.datapath, f"COCO_val2014_{img_id:012d}.jpg")

        try:
            entry["response"] = generate_response(model, processor, args, image_path, q)
            buffer.append(entry)
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at {img_id}, clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            traceback.print_exc()
            logger.error(f"⚠️ Failed {img_id}: {e}")
            errors.append(img_id)

        if len(buffer) >= save_every:
            _safe_write_json(output_json, processed + buffer)
            processed += buffer
            buffer.clear()

    if buffer:
        _safe_write_json(output_json, processed + buffer)

    print_run_chair_summary(start, total, output_json)
    logger.success(f"✅ Completed {len(processed)} / {total}. Errors: {len(errors)}")


def _safe_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ============================================
# 🧩 4. Entry Point
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--datapath", default="/mnt/disks/extra-disk/datasets/coco2014/val2014"
    )
    parser.add_argument("--method", default="greedy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--early_exit_layers", type=int, default=10)
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    # Setup logging and output dirs
    setup_logger(debug=args.debug, silent=args.silent)
    repo_owner, model_name = args.model_id.split("/")
    args.output = (
        args.output
        or f"./opera_log/{repo_owner}/{model_name}/{args.method}/responses.json"
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print_run_chair_header(args, args.output)
    model, processor = load_model(args.model_id, args.device)
    process_json(model, processor, args, args.output)
