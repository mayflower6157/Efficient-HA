import gc
import os
import argparse
import json
import numpy as np
import torch
import re
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    CLIPImageProcessor,
)  # , AutoModelForVision2Seq,CLIPImageProcessor
from qwen_vl_utils import process_vision_info
import time
import tqdm
from utils.logger_utils import print_run_header, print_run_summary
from utils.vcd_add_noise import add_diffusion_noise, add_diffusion_noise_pil

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from PIL import Image, ImageOps
from utils.vcd_add_noise import add_diffusion_noise
from utils.vcd_sample import evolve_vcd_sampling
from utils.deco_greedy import evolve_deco_greedy


def load_model(model_id, args):
    """Load the model and processor."""
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map=args.device,
    )

    model.eval()
    return model, processor


def get_response(model, processor, args, image_path, question):
    # For Qwen
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": (question)},
            ],
        }
    ]

    # Prepare inputs
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
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        if args.method == "greedy":
            generated_ids = model.generate(
                **inputs, max_new_tokens=args.max_tokens, do_sample=False
            )
        elif args.method == "beam":
            generated_ids = model.generate(
                **inputs, max_new_tokens=args.max_tokens, num_beams=5
            )

        elif args.method == "dola":
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                custom_generate="transformers-community/dola",
                dola_layers=[25, 35],
                do_sample=False,
                repetition_penalty=1.2,
                trust_remote_code=True,
            )
        elif args.method == "deco":
            evolve_deco_greedy()

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                top_p=None,
                top_k=None,
                do_sample=False,
                alpha=0.6,
                threshold_top_p=0.9,
                threshold_top_k=20,
                early_exit_layers=[i for i in range(25, 35)],
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

            generated_ids = generated_ids.sequences

        elif args.method == "vcd":
            evolve_vcd_sampling()

            inputs_cd = inputs.copy()
            inputs_cd["pixel_values"] = add_diffusion_noise(
                inputs["pixel_values"], args.noise_step
            )

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                pixel_values_cd=(inputs_cd["pixel_values"].unsqueeze(0).half().cuda()),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                do_sample=True,
            )
        else:
            raise ValueError(f"Unknown generation method: {args.method}")
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    return output_text


def process_json(model, processor, args, output_json):
    image_ids = []
    with open(
        "/home/mayflower/Efficient-HA/opera_log/llava-1.5/greedy.jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        for line in f.readlines():
            json_data = json.loads(line)
            json_data.pop("caption")
            image_ids.append(json_data)

    total_samples = len(image_ids)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    if not os.path.exists(output_json):
        with open(output_json, "w") as f:
            json.dump([], f)
    with open(output_json, "r") as f:

        current_data = json.load(f)
    processed_idx = [item["image_id"] for item in current_data]

    question = "Describe this image in detail."

    error_id = []

    start_time = time.time()

    for idx, line in enumerate(
        tqdm(image_ids, total=total_samples, desc="Processing", unit="img")
    ):
        if idx in processed_idx:
            continue

        image_path = "COCO_val2014_" + str(line["image_id"]).zfill(12) + ".jpg"
        image_path = os.path.join(args.datapath, image_path)

        response = get_response(model, processor, args, image_path, question)
        torch.cuda.empty_cache()

        line["response"] = response

        # tqdm handles ETA + progress bar, so no manual print needed
        with open(output_json, "r") as f:
            current_data = json.load(f)

        current_data.append(line)

        with open(output_json, "w") as f:
            json.dump(current_data, f, indent=2)

    print(error_id)

    print_run_summary(start_time, total_samples, output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to store model responses (auto-set if not provided)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to the model",
    )

    parser.add_argument(
        "--datapath",
        type=str,
        default="/mnt/disks/extra-disk/datasets/coco2014/val2014",
        help="Path to the data",
    )
    parser.add_argument("--method", type=str, default="greedy")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    # Auto-generate output path if not provided
    if args.output is None:
        repo_owner, model_name = args.model_id.split("/")
        base_dir = f"./opera_log/{repo_owner}/{model_name}/{args.method}"
        os.makedirs(base_dir, exist_ok=True)

        args.output = f"{base_dir}/responses.json"

    print_run_header(args, args.output)

    model, processor = load_model(args.model_id, args)

    process_json(model, processor, args, args.output)
