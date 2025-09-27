import gc
import os
import argparse
import json
import numpy as np
import torch
import re
from transformers import AutoProcessor, AutoModelForVision2Seq,CLIPImageProcessor
from qwen_vl_utils import process_vision_info
import time
from utils.vcd_add_noise import add_diffusion_noise,add_diffusion_noise_pil
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from PIL import Image, ImageOps
from utils.vcd_add_noise import add_diffusion_noise
from utils.vcd_sample import evolve_vcd_sampling
from utils.deco_greedy import evolve_deco_greedy


def load_model(model_id,args):
    """Load the model and processor."""
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                              min_pixels=min_pixels, max_pixels=max_pixels)

    model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    dtype='auto',
    trust_remote_code=True,
    device_map=args.device
)


    model.eval()
    return model, processor




def get_response(model, processor,args, image_path, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": (
                    question
                )},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        elif args.method == "beam":
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, num_beams=5
                                           )
            
        elif args.method == "dola":
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, custom_generate="transformers-community/dola",dola_layers=[25,35], do_sample=False,repetition_penalty=1.2,trust_remote_code=True)
        elif args.method == "deco":
            evolve_deco_greedy()

            generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, top_p=None, top_k=None, do_sample=False, alpha=0.6, threshold_top_p=0.9, threshold_top_k=20,
                                           early_exit_layers=[i for i in range(25,35)], return_dict_in_generate=True,
                    output_hidden_states=True)
            
            generated_ids = generated_ids.sequences

        elif args.method == "vcd":
            evolve_vcd_sampling()

            inputs_cd= inputs.copy()
            inputs_cd['pixel_values'] = add_diffusion_noise(inputs['pixel_values'], args.noise_step)


            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                pixel_values_cd=(inputs_cd['pixel_values'].unsqueeze(0).half().cuda()),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                do_sample=True)
        else:
            raise ValueError(f"Unknown generation method: {args.method}")
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text


def process_json(model, processor, args, output_json):

    with open('AMBER/data/query/query_all.json', "r", encoding="utf-8") as f:
        json_data = json.load(f) 


    total_samples = len(json_data)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    ans_file = open(output_json, "w")

    for idx, line in enumerate(json_data):

        image_path = line['image']
        question = line['query']
        image_path = os.path.join(args.datapath, image_path)
        idx = line['id']



        response = get_response(model, processor,args, image_path, question)

        torch.cuda.empty_cache()

        line['response'] = response


        print(f"Processed sample {idx + 1}/{total_samples}: {response[:50]}...")



        res_dict = {"id": idx,"response": response, "prompt": question, "image": line['image']}
        ans_file.write(json.dumps(res_dict) + "\n")
        ans_file.flush()


    print(f"All results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default='/home/li0007xu/Reasoning/test/output_deco.jsonl',
                        help='Output file to store model responses')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help='Path to the model')
    
    parser.add_argument('--datapath', type=str, default="/home/li0007xu/P1/TTAH/Qwen2.5-VL/amberdata/image/",
                        help='Path to the data')
    parser.add_argument('--method', type=str, default="vcd")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--max_tokens', type=int, default=64)
    args = parser.parse_args()

    model, processor = load_model(args.model_id,args)

    process_json(model, processor,args, args.output)

