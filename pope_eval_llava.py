import argparse
import os
import random
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from pope_loader import POPEDataSet
from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *



POPE_PATH = {
    "random": "pope_style/coco_pope_random.json",
    "popular": "pope_style/coco_pope_popular.json",
    "adversarial": "pope_style/coco_pope_adversarial.json",
}

def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model",default='llava')
    parser.add_argument("--pope-type", type=str, help="random",default='random')
    # parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--log_path", type=str, default="/home/li0007xu/P1/OPERA/testlog/", help="log path")
    parser.add_argument("--data_path", type=str, default="COCO_2014/val2014/", help="data path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="num workers")
    parser.add_argument("--decoding", type=str,default='oursbeamsample')
    parser.add_argument("--beam", type=int)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--ours_alpha", type=float, default=0.6)
    parser.add_argument("--ours_threshold_top_p", type=float, default=0.9)
    parser.add_argument("--ours_threshold_top_k_small", type=int, default=50)
    parser.add_argument("--ours_threshold_top_k", type=int, default=20)

    parser.add_argument(
        "--style",
        type=str,
        # default="None",
        default="original",
        # default="/home/li0007xu/OPERA/val2014/",
        help="style image path",
    )

    args = parser.parse_args()
    return args

def setup_seeds(seed):
    import random
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


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
    with open(os.path.join(base_dir, 'POPE_style_{}_decoding_{}_type_{}.jsonl'.format(args.style,args.decoding,args.pope_type)), "a") as f:
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




def main():

    args = parse_args()

    style=args.style

    if style == "cartoon":
        args.data_path = "/home/DTC_SSD/Stylev1.2/Cartoon/"

    elif style == "game":
        args.data_path = "/home/DTC_SSD/Stylev1.2/Game/"

    elif style == "graffiti":
        args.data_path = "/home/DTC_SSD/Stylev1.2/Graffiti/"

    elif style == "painting":
        args.data_path = "/home/DTC_SSD/Stylev1.2/Painting/"

    elif style == "sketch":
        args.data_path = "/home/DTC_SSD/Stylev1.2/Sketch/"

    elif style == "original":
        args.data_path = "/home/DTC_SSD/Stylev1.2/Original/"


    args.pope_path = POPE_PATH[args.pope_type]


    setup_seeds(0)


    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # load pope data
    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path, 
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    print ("load data finished")
    base_dir = os.path.join(args.log_path, "Pope", args.model,args.decoding)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print("Start eval...")
    pred_list, pred_list_s, label_list = [], [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image_path = data["image_path"][0]
        qu = data["query"][0]
        label = data["label"]
        label_list = label_list + list(label)
        conversation = [
    {
        "role": "user",
        "content": [
                            {
                    "type": "image",
                    "image":  image_path,

                },
            {"type": "text", "text": qu},
        ],
    },
]
        inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)


        label = torch.Tensor(label).to(model.device)
        if args.decoding == 'oursbeamsample':
            beams=3
            use_ours=True
            output_hidden_states=True
            key_position = {
                            "image_start": inputs['input_ids'].tolist()[0].index(32000), 
                            "image_end": inputs['input_ids'].tolist()[0].index(32000)+24*24, 
                        }
#            key_position = {
                        #     "image_start": inputs['input_ids'].tolist()[0].index(32000), 
                        #     "image_end": inputs['input_ids'].tolist()[0].index(32000)+24*24, 
                        # }
            print(key_position)
            inputs = inputs.to(model.device)
            early_exit_layers=[i for i in range(20, 29)]
            model=model.eval()
            with torch.inference_mode():
                with torch.no_grad():  
                    generated_ids = model.generate(**inputs, max_new_tokens=10,num_beams=beams,use_ours=use_ours,output_hidden_states=output_hidden_states,return_dict_in_generate=True,ours_layers=early_exit_layers,key_position=key_position,ours_alpha = args.ours_alpha,
                        ours_threshold_top_k_small=args.ours_threshold_top_k_small,
                        ours_threshold_top_p=args.ours_threshold_top_p,
                        ours_threshold_top_k=args.ours_threshold_top_k) 
                    generated_ids=generated_ids.sequences
                    output_text=processor.batch_decode(generated_ids[:, len(inputs['input_ids'][0]):], skip_special_tokens=True)
                    print(output_text[0])
                    pred_list = recorder(output_text[0], pred_list)

        elif args.decoding == 'deco':
            early_exit_layers=[i for i in range(20, 29)]
            beams=3
            alpha = 0.9
            threshold_top_p=0.9
            threshold_top_k=20
            use_deco=True
            use_ours=False
            inputs = inputs.to(model.device)
            model=model.eval()
            with torch.inference_mode():
                with torch.no_grad():  
                    generated_ids = model.generate(**inputs, max_new_tokens=10,num_beams=beams,use_deco=use_deco,output_hidden_states=True,alpha=alpha,return_dict_in_generate=True,threshold_top_p=threshold_top_p,threshold_top_k=threshold_top_k,early_exit_layers=early_exit_layers) 
                    generated_ids=generated_ids.sequences
                    output_text=processor.batch_decode(generated_ids[:, len(inputs['input_ids'][0]):], skip_special_tokens=True)
                    print(output_text)
                    pred_list = recorder(output_text[0], pred_list)

        elif args.decoding == 'greedy':

            beams=1

            use_deco=False
            use_ours=False
            inputs = inputs.to(model.device)
            model=model.eval()
            with torch.inference_mode():
                with torch.no_grad():  
                    generated_ids = model.generate(**inputs, max_new_tokens=10,num_beams=beams,use_deco=use_deco,output_hidden_states=True,return_dict_in_generate=True) 
                    generated_ids=generated_ids.sequences
                    output_text=processor.batch_decode(generated_ids[:, len(inputs['input_ids'][0]):], skip_special_tokens=True)
                    print(output_text)
                    pred_list = recorder(output_text[0], pred_list)
        elif args.decoding == 'dola':

            beams=1
            early_exit_layers=[i for i in range(20, 29)]
            use_deco=False
            use_ours=False
            inputs = inputs.to(model.device)
            model=model.eval()
            with torch.inference_mode():
                with torch.no_grad():  
                    generated_ids = model.generate(**inputs, max_new_tokens=10,num_beams=beams,dola_layers=early_exit_layers,output_hidden_states=True,return_dict_in_generate=True,do_sample=False) 
                    generated_ids=generated_ids.sequences
                    output_text=processor.batch_decode(generated_ids[:, len(inputs['input_ids'][0]):], skip_special_tokens=True)
                    print(output_text)
                    pred_list = recorder(output_text[0], pred_list)
        elif args.decoding == 'beam':

            beams=3

            use_deco=False
            use_ours=False
            inputs = inputs.to(model.device)
            model=model.eval()
            with torch.inference_mode():
                with torch.no_grad():  
                    generated_ids = model.generate(**inputs, max_new_tokens=10,num_beams=beams,use_deco=use_deco,output_hidden_states=True,return_dict_in_generate=True) 
                    generated_ids=generated_ids.sequences
                    output_text=processor.batch_decode(generated_ids[:, len(inputs['input_ids'][0]):], skip_special_tokens=True)
                    print(output_text)
                    pred_list = recorder(output_text[0], pred_list)
    if len(pred_list) != 0:
        print_acc(pred_list, label_list,args,base_dir)
    if len(pred_list_s) != 0:
        print_acc(pred_list_s, label_list,args,base_dir)









if __name__ == "__main__":
    main()
