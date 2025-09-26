import gc
import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
from transformers import AutoProcessor, AutoModelForVision2Seq,CLIPImageProcessor
from qwen_vl_utils import process_vision_info
import time
from vcd_utils.vcd_add_noise import add_diffusion_noise,add_diffusion_noise_pil
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from PIL import Image, ImageOps
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.deco_greedy import evolve_deco_greedy
from vcd_utils.ours import evolve_ours_greedy
def find_between_sequences(tokens, start_seq, end_seq, from_index=0):
    """在 tokens 中寻找 start_seq 与 end_seq 之间的区间，返回 (content_start, content_end)（end 为不含）"""
    n = len(tokens)
    m1, m2 = len(start_seq), len(end_seq)
    
    def find_seq(seq, start):
        for i in range(start, n - len(seq) + 1):
            if tokens[i:i+len(seq)] == seq:
                return i
        return -1
    
    s = find_seq(start_seq, from_index)
    if s == -1:
        return -1, -1
    e = find_seq(end_seq, s + m1)
    if e == -1:
        return -1, -1
    return s + m1, e
def load_model(model_id,args):
    """Load the model and processor."""
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                              min_pixels=min_pixels, max_pixels=max_pixels)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_id,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    # ).to(args.device).eval()
    model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    attn_implementation="eager",
    device_map=args.device
)
    model.eval()
    return model, processor


def extract_thinking(response):
    """Extracts the thinking part from response text, including the <think> tags."""
    match = re.search(r"(<think>.*?</think>)", response, re.DOTALL)
    if match:
        thinking_text = match.group(1).strip()
        return thinking_text, len(processor.tokenizer(thinking_text, return_tensors='pt')['input_ids'][0])
    return "", -1



def get_response(model, processor,args, image_path, question,neg_image_path=None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": (
                    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags.\n" + question
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
            generated_ids = model.generate(**inputs, max_new_tokens=4096)
        elif args.method == "beam":
            generated_ids = model.generate(**inputs, max_new_tokens=4096, num_beams=5
                                           )
            
        elif args.method == "dola":
            generated_ids = model.generate(**inputs, max_new_tokens=4096, dola_layers=[16,18,20,22,24,26,28,30,32], do_sample=False,repetition_penalty=1.2)
        elif args.method == "deco":
            evolve_deco_greedy()

            generated_ids = model.generate(**inputs, max_new_tokens=4096,top_p=None,top_k=None, do_sample=False, alpha=0.6, threshold_top_p=0.9, threshold_top_k=20,
                                           early_exit_layers=[i for i in range(20,29)],return_dict_in_generate=True,
                    output_hidden_states=True)
            
            generated_ids = generated_ids.sequences
        elif args.method=='ours':


            image_inputs_po=[add_diffusion_noise_pil(image_inputs[0], 500)]

    
            

            inputs['positive_images']=processor(text=[text],images=image_inputs_po,videos=video_inputs, padding=True,return_tensors="pt").pixel_values.to(model.device)

            inputs = inputs.to(model.device)

            evolve_ours_greedy()
            tokenizer = processor.tokenizer 
            len_question=len(processor(text=[question])['input_ids'][0])
            generated_ids= model.generate(**inputs, max_new_tokens=4096, 
                        output_hidden_states=True,
                        output_attentions=True,
                        len_question=len_question,

                        tokenizer_r=tokenizer,
                        return_dict_in_generate=True)
            # generated_ids= model.generate(**inputs, max_new_tokens=2048, 
            #                             output_hidden_states=True,
            #                             output_attentions=True,
            #                             len_question=len_question,
            #                             # tokenizer_r=tokenizer,
            #                             return_dict_in_generate=True,
            #                             top_p=None,top_k=None, do_sample=False, alpha=0.6, threshold_top_p=0.9, threshold_top_k=20,
            #                                     early_exit_layers=[i for i in range(20,28)])
            generated_ids = generated_ids.sequences
        #     generated_ids_trimmed = [
        #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)
        #     ]
        # # [all_max_img_token_att,all_max_ques_token_att,all_sum_img_token_att,all_sum_ques_token_att]
        #     output_text = processor.batch_decode(
        #         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        #     )[0]
            

            
        #     s=[]
        #     for i in range(len(generated_ids_trimmed[0])):
        #         id_text={int(generated_ids_trimmed[0][i]):processor.decode(generated_ids_trimmed[0][i])}

        #         s.append(id_text)
        #     tokens = [list(d.values())[0] for d in s]
        #     think_start_seq = ["think" ]
        #     think_end_seq   = [ "think"]

        #     answer_start_seq = ["answer"]
        #     answer_end_seq   = [ "answer"]

        #     think_range = find_between_sequences(tokens, think_start_seq, think_end_seq)
        #     answer_range = find_between_sequences(tokens, answer_start_seq, answer_end_seq)


        #     inputs=inputs.to("cpu")
        #     think_start=think_range[0]
        #     if think_start==-1:
        #         think_start=0
        #     think_end=min(think_range[1],answer_range[0]) if answer_range[0]!=-1 else think_range[1]
        #     if think_end==-1:
        #         think_end=answer_range[0]
        #     think=[think_start+len(inputs.input_ids[0])-2,think_end+len(inputs.input_ids[0])+2]
            
        #     think_text=processor.batch_decode(
        #         generated_ids.sequences[:,think[0]:think[1]], skip_special_tokens=False, clean_up_tokenization_spaces=False
        #     )[0]
            
        #     think_text=text+think_text
        #     think_more_inputs = processor(
        #         text=[think_text],
        #         images=image_inputs,
        #         videos=video_inputs,
        #         padding=True,
        #         return_tensors="pt",
        #     )
        #     think_more_inputs = think_more_inputs.to(model.device)

        #     generated_ids,all_atten,index = model.generate(**think_more_inputs, max_new_tokens=4096, 
        #                                 output_hidden_states=False,
        #                                 output_attentions=True,
        #                                 len_question=len_question,
        #                                 think_range=think,
        #                                 tokenizer_r=tokenizer,
        #                                 return_dict_in_generate=True)



        elif args.method == "vcd":
            evolve_vcd_sampling()

            inputs_cd= inputs.copy()
            inputs_cd['pixel_values'] = add_diffusion_noise(inputs['pixel_values'], args.noise_step)


            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                pixel_values_cd=(inputs_cd['pixel_values'].unsqueeze(0).half().cuda()),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
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


def process_json(model, processor, args,input_json, output_json):

    with open('/home/li0007xu/P1/TTAH/Qwen2.5-VL/amberdata/AMBER/data/query/query_all.json', "r", encoding="utf-8") as f:
            json_data = json.load(f) 

            neg_json_data = json.load(open('/home/li0007xu/Reasoning/RH-Bench/reason_data.json', 'r'))

    total_samples = len(json_data)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    if not os.path.exists(output_json):
        with open(output_json, 'w') as f:
            json.dump([], f)
    with open(output_json, 'r') as f:

        current_data=json.load(f) 
    processed_idx=[item['id'] for item in current_data]


    thinking_lengths = []
    responses_data = []
    i=1
    # 遍历并处理每个样本
    error_id=[]
    for idx, line in enumerate(json_data):
        if idx in processed_idx:
            continue
        image_path = line['image']
        neg_image_path = None
        if args.RH == 'reason':

            image_path = os.path.join('/home/li0007xu/P1/TTAH/Qwen2.5-VL/amberdata/image/', image_path)
        else:
            image_path = os.path.join('/home/li0007xu/P1/TTAH/Qwen2.5-VL/amberdata/image/', image_path)

        question = line['query']

        if args.method=='ours':

            response = get_response(model, processor,args, image_path, question,neg_image_path)
        else:
            response = get_response(model, processor,args, image_path, question)

        torch.cuda.empty_cache()
        thinking_part, thinking_length = extract_thinking(response)
        thinking_lengths.append(thinking_length)

        line['response'] = response
        line['thinking'] = thinking_part
        line['thinking_length'] = thinking_length

        print(f"Processed sample {idx + 1}/{total_samples}: {response[:50]}...")

        with open(output_json, 'r') as f:
            current_data = json.load(f)

        current_data.append(line)

        with open(output_json, 'w') as f:
            json.dump(current_data, f, indent=2)

    print(error_id)



    print(f"All results saved to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='/home/li0007xu/Reasoning/RH-Bench/reason_data.json',
                        help='Template file containing images and questions')
    parser.add_argument('--output', type=str,
                        default='/home/li0007xu/Reasoning/test/output_ours.json',
                        help='Output file to store model responses')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help='Path to the model')

    parser.add_argument('--RH', type=str, default='hallu',
                        help='Root directory for images')
    parser.add_argument('--method', type=str, default="ours")
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument('--device', type=str, default="cuda:5")
    args = parser.parse_args()

    model, processor = load_model(args.model_id,args)

    process_json(model, processor,args, args.input, args.output)

