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
from tqdm import tqdm
from utils.vcd_add_noise import add_diffusion_noise
from utils.vcd_sample import evolve_vcd_sampling
from utils.deco_greedy import evolve_deco_greedy
from pope_loader import POPEDataSet
POPE_PATH = {
    "random": "pope_coco/coco_pope_random.json",
    "popular": "pope_coco/coco_pope_popular.json",
    "adversarial": "pope_coco/coco_pope_adversarial.json",
}
def print_acc(pred_list, label_list,args,base_dir):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    with open(os.path.join(base_dir, 'POPE__type_{}.jsonl'.format(args.pope_type)), "a") as f:
        json.dump({
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Yes_ratio": yes_ratio,
        }, f)
        f.write('\n')
    
        

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    line=out

    line = line.replace('.', '')
    line = line.replace(',', '')
    words = line.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        pred_list.append(0)
    else:
        pred_list.append(1)
    
    return pred_list

def load_model(model_id,args):
    """Load the model and processor."""
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                              min_pixels=min_pixels, max_pixels=max_pixels)

    model = AutoModelForVision2Seq.from_pretrained(
    model_id,
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


def process_json(model, processor, args, output):
    args.pope_path = POPE_PATH[args.pope_type]
    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.datapath, 
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2,
        drop_last=False
    )

    total_samples = len(pope_dataset)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    pred_list, pred_list_s, label_list = [], [], []

    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        if batch_id ==5:
            break
        image_path = data["image_path"][0]
        qu = data["query"][0]
        label = data["label"]
        label_list = label_list + list(label)

        label = torch.Tensor(label).to(model.device)
        
        response = get_response(model, processor,args, image_path, qu)

        torch.cuda.empty_cache()
        pred_list = recorder(response, pred_list)


    if len(pred_list) != 0:
        print_acc(pred_list, label_list,args,args.output)
    if len(pred_list_s) != 0:
        print_acc(pred_list_s, label_list,args,args.output)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default='/home/li0007xu/Reasoning/test',
                        help='Output file to store model responses')

    parser.add_argument("--pope-type", type=str, help="random",default='random', choices=['random', 'popular', 'adversarial'])
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help='Path to the model')
    
    parser.add_argument('--datapath', type=str, default="/home/li0007xu/EH/Efficient-HA/val2014",
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

