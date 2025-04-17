import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import math
from PIL import Image
import transformers
import time

from model import GeckoForConditionalGeneration, GeckoConfig, GeckoProcessor
from model.conversation import conv_templates


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_crop_size_from_name(model_name):
    vision_encoder = model_name.split('/')[-1].split('-')[-2]
    if vision_encoder == 'siglip':
        crop_size = 384
    elif vision_encoder == 'clip':
        crop_size = 336
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return crop_size

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def eval_model(args):
    disable_torch_init()
    config = GeckoConfig.from_pretrained(args.model_name_or_path, 
                                         topk=args.topk, 
                                         visualize_patches=args.visualize_top_patches, 
                                         visualize_topk_patches=args.visualize_top_patches,
                                         keyword_criteria=args.keyword_criteria,
                                         positional_information=args.positional_information,
                                         vision_feature_select_strategy=args.vision_feature_select_strategy,
                                         patch_picking_strategy=args.patch_picking_strategy)
    crop_size = get_crop_size_from_name(args.model_name_or_path)
    processor = GeckoProcessor.from_pretrained(args.model_name_or_path, config=config, use_keyword=True, cropping_method=args.cropping_method, crop_size=crop_size)
    model = GeckoForConditionalGeneration.from_pretrained(
        args.model_name_or_path, config=config, torch_dtype=args.torch_dtype, 
        attn_implementation=args.attn_implementation, device_map=args.device_map)
    model.load_text_encoder(processor=processor)

    model_name = args.model_name_or_path.split("/")[-1]
    eos_token_id = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    start_time = time.time()
    for i, line in enumerate(tqdm(questions)):
        idx = line['question_id']
        qs = line['text']

        image_files = line.get("images", line.get("image"))
        image_files = [image_files] if isinstance(image_files, str) else image_files
        images = [Image.open(os.path.join(args.image_folder, image_file)) for image_file in image_files]
        image_prompt = '<image>' + '\n'
        qs = image_prompt * len(images) + qs

        conv = conv_templates[args.conv_format].copy()
        conv.messages = []

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], "")

        # keyword = line['keywords']
        keyword_prompt = conv.generate_keyword_prompt(qs.split('\n')[len(images)])
        prompt = conv.get_prompt()

        inputs = processor(images=images, text=prompt, keywords_text=keyword_prompt, return_tensors="pt")

        for k, v in inputs.items():
            if v is not None:
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)
                elif isinstance(v, list):
                    if k == 'coords':
                        continue
                    inputs[k] = [x.to(model.device) for x in v]
                elif isinstance(v, transformers.tokenization_utils_base.BatchEncoding) or isinstance(v, dict):
                    for key, value in v.items():
                        if value is not None:
                            if isinstance(value, list):
                                inputs[k][key] = [x.to(model.device) for x in value]
                            else:
                                inputs[k][key] = value.to(model.device)
                else:
                    raise ValueError(f"Invalid input type: {type(v)}")
                
        with torch.inference_mode():
            output_ids = model.generate(**inputs,
                                        eos_token_id=eos_token_id,
                                        do_sample=True if args.temperature > 0 else False,
                                        temperature=args.temperature,
                                        max_new_tokens=1024,
                                        use_cache=True)[0]

        generated_ids = output_ids[inputs["input_ids"].shape[-1]:]
        outputs = processor.decode(generated_ids, skip_special_tokens=True)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                            "prompt": qs,
                            "text": outputs,
                            "answer_id": ans_id,
                            "model_id": model_name,
                            "metadata": {}}) + "\n")
        
        ans_file.flush()
    ans_file.close()
    print(f"Time taken: {time.time() - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--torch_dtype", type=str, default=torch.float16)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--device_map", type=str, default="cuda")
    parser.add_argument("--conv_format", type=str, default="llama_3")
    parser.add_argument("--visualize_top_patches", type=bool, default=False)
    parser.add_argument("--vision_feature_select_strategy", type=str, default="cls")
    parser.add_argument("--patch_picking_strategy", type=str, default="across_layers")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--cropping_method", type=str, default="dynamic")
    parser.add_argument("--crop_size", type=int, default=336)
    parser.add_argument("--keyword_criteria", type=str, default="description")
    parser.add_argument("--positional_information", type=str, default="explicit")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    print(args)
    eval_model(args)