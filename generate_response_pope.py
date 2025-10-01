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


from PIL import Image, ImageOps
from tqdm import tqdm
from utils.vcd_add_noise import add_diffusion_noise
from utils.vcd_sample import evolve_vcd_sampling
from utils.deco_greedy import evolve_deco_greedy
from utils.logger_utils import print_run_pope_header, print_run_pope_summary
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


def get_num_layers(model):
    # 1. Direct field (LLaMA/Qwen style)
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers

    # 2. Gemma-3 style (nested inside text_config)
    elif hasattr(model.config, "text_config") and hasattr(
        model.config.text_config, "num_hidden_layers"
    ):
        return model.config.text_config.num_hidden_layers

    # 3. Decoder layers (OPT, LLaMA, Gemma, etc.)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)

    # 4. Encoder layers (T5, BART)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return len(model.encoder.layers)

    raise ValueError("Could not auto-detect number of layers for this model.")


def get_early_exit_layers(model, n):
    num_layers = get_num_layers(model)
    max_layer_index = num_layers  # last hidden state index = num_layers
    early_exit_layers = list(range(max(1, num_layers - (n - 1)), max_layer_index + 1))
    return early_exit_layers


def print_acc(pred_list, label_list, args, base_dir):
    # Ensure directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(label_list, pred_list, labels=[1, 0])
    TP, FP, FN, TN = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Accuracy
    acc = accuracy_score(label_list, pred_list)

    # Classification report (precision, recall, f1)
    report = classification_report(
        label_list, pred_list, target_names=["Negative", "Positive"], digits=4
    )

    print("Confusion Matrix (labels: [Positive=1, Negative=0])")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {acc:.4f}")

    # Save metrics
    with open(
        os.path.join(base_dir, f"POPE_type_{args.pope_type}_{args.method}.jsonl"), "a"
    ) as f:
        json.dump(
            {
                "ConfusionMatrix": cm.tolist(),
                "Accuracy": acc,
                "Report": classification_report(
                    label_list, pred_list, output_dict=True
                ),
            },
            f,
        )
        f.write("\n")


def recorder(out, pred_list):
    text = re.sub(r"[.,]", "", out).lower()
    if any(w in text.split() for w in ["no", "not"]) or "n't" in text:
        pred_list.append(0)
    else:
        pred_list.append(1)
    return pred_list


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


def prepare_inputs(model, processor, image_path, question):
    """Build model-ready inputs from image + text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": q},
            ],
        }
        for img, q in zip(image_path, question)
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
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
        evolve_vcd_sampling()  # side-effect initialization
        inputs_cd = copy.deepcopy(inputs)
        inputs_cd["pixel_values"] = add_diffusion_noise(
            inputs["pixel_values"], args.noise_step
        )
        return model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            pixel_values_cd=(inputs_cd["pixel_values"].unsqueeze(0).half().cuda()),
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
            do_sample=True,
        )

    raise ValueError(f"Unknown generation method: {args.method}")


def decode_output(processor, inputs, generated_ids):
    """Trim input tokens and decode generated sequence into text."""
    trimmed = [
        out_ids[len(in_ids) :] if len(out_ids) > len(in_ids) else out_ids
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def get_response(model, processor, args, image_path, question):
    """High-level wrapper: prepare → generate → decode."""
    inputs = prepare_inputs(model, processor, image_path, question)
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
        num_workers=2,
        drop_last=False,
    )

    total_samples = len(pope_dataset)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    pred_list, label_list = [], []
    start_time = time.time()
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        # Loop over items in the batch
        responses = get_response(
            model, processor, args, data["image_path"], data["query"]
        )
        for resp, label in zip(responses, data["label"]):
            pred_list = recorder(resp, pred_list)
            label_list.append(int(label))

        if batch_id % 10 == 0:
            torch.cuda.empty_cache()

    if len(pred_list) != 0:
        print_acc(pred_list, label_list, args, args.output)

    # Print run summary
    print_run_pope_summary(start_time, total_samples, args.output)


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
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--early_exit_layers", type=int, default=10)
    args = parser.parse_args()

    # Print run header
    print_run_pope_header(args, args.output)

    model, processor = load_model(args.model_id, args)

    process_json(model, processor, args, args.output)
