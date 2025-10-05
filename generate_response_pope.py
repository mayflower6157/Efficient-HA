import copy
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
)  # , AutoModelForVision2Seqrocessor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from qwen_vl_utils import process_vision_info
import time
from utils.vcd_add_noise import add_diffusion_noise, add_diffusion_noise_pil

# === Utilities ===
from utils.logger_utils import (
    print_run_pope_header,
    print_run_pope_summary,
    setup_logger,
)
from PIL import Image, ImageOps
from tqdm import tqdm
from utils.vcd_add_noise import add_diffusion_noise
from utils.vcd_sample import evolve_vcd_sampling
from utils.deco_greedy import evolve_deco_greedy, get_early_exit_layers
from pope_loader import POPEDataSet

POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_acc(pred_list, label_list, args, base_dir):
    """
    Compute and print POPE metrics, and save them as JSONL files.
    Compatible with the existing evaluation structure.
    """
    os.makedirs(base_dir, exist_ok=True)

    # === Compute metrics ===
    cm = confusion_matrix(label_list, pred_list, labels=[1, 0])
    TP, FP, FN, TN = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = accuracy_score(label_list, pred_list)
    report_dict = classification_report(label_list, pred_list, output_dict=True)
    report_text = classification_report(
        label_list, pred_list, target_names=["Negative", "Positive"], digits=4
    )

    # === Print metrics to console ===
    print("\n==================== POPE Evaluation ====================")
    print(f"POPE Type: {args.pope_type} | Method: {args.method}")
    print("----------------------------------------------------------")
    print("Confusion Matrix (labels: [Positive=1, Negative=0])")
    print(cm)
    print("\nClassification Report:")
    print(report_text)
    print(f"Accuracy: {acc:.4f}")
    print("==========================================================")

    # === Save metrics to JSONL ===
    metric_path = os.path.join(
        base_dir, f"POPE_type_{args.pope_type}_{args.method}_metric.jsonl"
    )

    with open(metric_path, "a") as f:
        json.dump(
            {
                "POPE_Type": args.pope_type,
                "Method": args.method,
                "ConfusionMatrix": cm.tolist(),
                "Accuracy": acc,
                "Report": report_dict,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"✅ Metrics appended to: {metric_path}\n")


def recorder(out):
    text = out.lower()
    if re.search(r"\b(?:no|not|n't)\b", text):
        return 0
    else:
        return 1


def load_model(model_id, args):
    """Load the model and processor."""
    try:
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # ✅ Fix: decoder-only models like Qwen expect left-padding
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map=args.device,
        )

        model.eval()
        return model, processor
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        raise


def format_images_for_gemma(images):
    """Format images Gemma-3 style: [[img1], [img2], ...]."""
    return [[img] for img in images]


def prepare_inputs(model, processor, image_paths, questions):
    """Build model-ready inputs from batches of images + text."""

    # Ensure inputs are lists for batch processing
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if isinstance(questions, str):
        questions = [questions]

    # Validate batch sizes match
    batch_size = len(image_paths)
    if len(questions) != batch_size:
        raise ValueError(
            f"Batch size mismatch: {len(image_paths)} images vs {len(questions)} questions"
        )

    # Build messages for each item in the batch
    messages = []
    for img_path, question in zip(image_paths, questions):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question},
                ],
            }
        )

    # Create one text + image pair per sample
    # Apply chat template
    texts = [
        processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

    # 🚀 Instead of process_vision_info → pass image_paths directly
    model_type = getattr(model.config, "model_type", "").lower()
    if "gemma" in model_type:
        # Gemma expects list-of-lists [[img1],[img2],...]
        image_paths = format_images_for_gemma(image_paths)
    else:
        # Qwen, LLaVA etc. accept flat list [img1,img2,...]
        image_paths = image_paths

    inputs = processor(
        text=texts,
        images=image_paths,  # let processor handle batching
        padding=True,
        return_tensors="pt",
    )

    return inputs.to(model.device)


def generate_ids(model, inputs, args):
    """Generate token IDs using the chosen decoding method."""
    method = args.method.lower()

    if method == "greedy":
        return model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)

    if method == "beam":
        return model.generate(**inputs, max_new_tokens=args.max_tokens, num_beams=5)

    if method == "dola":
        early_exit_layers = get_early_exit_layers(model, args.early_exit_layers)
        return model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            custom_generate="transformers-community/dola",
            dola_layers=early_exit_layers,
            do_sample=False,
            repetition_penalty=1.2,
            trust_remote_code=True,
        )

    if method == "deco":
        evolve_deco_greedy()  # side-effect initialization
        early_exit_layers = get_early_exit_layers(model, args.early_exit_layers)
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            alpha=0.6,
            threshold_top_p=0.9,
            threshold_top_k=20,
            early_exit_layers=early_exit_layers,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        return out.sequences

    if method == "vcd":
        evolve_vcd_sampling()
        # Fix: Handle batched pixel_values correctly
        pixel_values_noisy = add_diffusion_noise(
            inputs["pixel_values"], args.noise_step
        )

        return model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            pixel_values_cd=pixel_values_noisy,  # Already in correct shape/dtype
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
            do_sample=True,
        )

    raise ValueError(f"Unknown generation method: {args.method}")


def decode_output(processor, inputs, generated_ids):
    """Trim input tokens and decode generated sequence into text."""
    # Handle attention_mask to get actual input lengths per sample
    input_lengths = inputs.attention_mask.sum(dim=1).tolist()

    trimmed = [
        out_ids[input_len:] if len(out_ids) > input_len else out_ids
        for input_len, out_ids in zip(input_lengths, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def get_response(model, processor, args, image_paths, questions):
    """High-level wrapper: prepare → generate → decode."""
    inputs = prepare_inputs(model, processor, image_paths, questions)
    with torch.inference_mode():
        generated_ids = generate_ids(model, inputs, args)
        return decode_output(processor, inputs, generated_ids)


def process_json(model, processor, args, output):
    args.pope_path = POPE_PATH[args.pope_type]
    pope_dataset = POPEDataSet(
        pope_path=args.pope_path,
        data_path=args.datapath,
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    total_samples = len(pope_dataset)
    # Fix: Handle both directory and file paths
    if output:
        if os.path.isdir(output) or output.endswith("/"):
            # If output is a directory, create the file inside it
            os.makedirs(output, exist_ok=True)
            base_dir = output
            detail_file = os.path.join(
                output, f"POPE_type_{args.pope_type}_{args.method}_detailed.jsonl"
            )
        else:
            # If output is a file path, use its directory
            os.makedirs(os.path.dirname(output), exist_ok=True)
            base_dir = os.path.dirname(output)
            detail_file = output.replace(".jsonl", "_detailed.jsonl")
    else:
        base_dir = "outputs"
        os.makedirs(base_dir, exist_ok=True)
        detail_file = os.path.join(
            base_dir, f"POPE_type_{args.pope_type}_{args.method}_detailed.jsonl"
        )

    pred_list, label_list = [], []
    detailed_results = []  # Store detailed predictions
    start_time = time.time()
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        # Loop over items in the batch
        responses = get_response(
            model, processor, args, data["image_path"], data["query"]
        )
        for resp, label, img_path, query in zip(
            responses, data["label"], data["image_path"], data["query"]
        ):
            pred = recorder(resp)
            pred_list.append(pred)
            label_list.append(int(label))

            # Store detailed result
            detailed_results.append(
                {
                    "image": os.path.basename(img_path),
                    "question": query,
                    "response": resp,
                    "prediction": pred,
                    "label": int(label),
                    "correct": pred == int(label),
                }
            )
        if batch_id % 5 == 0:
            torch.cuda.empty_cache()

    # Save detailed results
    with open(detail_file, "w") as f:
        for result in detailed_results:
            f.write(json.dumps(result) + "\n")

    if len(pred_list) != 0:
        print_acc(pred_list, label_list, args, args.output)

    # Print run summary
    print_run_pope_summary(start_time, total_samples, args.output)


def validate_args(args):
    """Validate argument combinations."""
    if args.method in ["dola", "deco"] and args.early_exit_layers < 1:
        raise ValueError(f"early_exit_layers must be >= 1 for {args.method}")

    if args.batch_size > 16 and args.method == "vcd":
        print("Warning: Large batch size with VCD may cause OOM. Consider reducing.")

    if not os.path.exists(args.datapath):
        raise FileNotFoundError(f"Data path not found: {args.datapath}")

    return args


if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to store model responses",
    )

    parser.add_argument(
        "--pope_type",
        type=str,
        help="random",
        default="random",
        choices=["random", "popular", "adversarial"],
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for DataLoader (depends on GPU memory)",
    )
    parser.add_argument("--method", type=str, default="vcd")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--early_exit_layers", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    args = validate_args(args)

    # Setup logging and output dirs
    setup_logger(debug=args.debug, silent=args.silent)
    # Print run header
    print_run_pope_header(args, args.output)

    model, processor = load_model(args.model_id, args)

    process_json(model, processor, args, args.output)
