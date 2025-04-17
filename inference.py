import argparse
import torch
import os
import json
from tqdm import tqdm
import math
from PIL import Image
import transformers
import time

from model import GeckoForConditionalGeneration, GeckoConfig, GeckoProcessor
from model.conversation import conv_templates

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def get_crop_size_from_name(model_name):
    vision_encoder = model_name.split('/')[-1].split('-')[-2]
    if vision_encoder == 'siglip':
        crop_size = 384
    elif vision_encoder == 'clip':
        crop_size = 336
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return crop_size

def inference(args):
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

    eos_token_id = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    qs = args.question
    if args.images:
        images = [Image.open(image) for image in args.images]
        image_prompt = '<image>' + '\n'
        qs = image_prompt * len(images) + qs
    else:
        images = None

    conv = conv_templates[args.conv_format].copy()
    conv.messages = []

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], "")

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

    print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model or model name, e.g., farrosalferro24/Gecko-Mantis-8B-siglip-llama3")
    parser.add_argument("--images", type=str, nargs='+', help="List of image paths to use as input")
    parser.add_argument("--torch_dtype", type=str, default=torch.float16)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--device_map", type=str, default="cuda")
    parser.add_argument("--conv_format", type=str, default="llama_3")
    parser.add_argument("--visualize_top_patches", type=bool, default=False, help="Whether to visualize the top patches")
    parser.add_argument("--vision_feature_select_strategy", type=str, default="cls", help="Strategy for selecting vision features. Options: 'cls', 'image_features")
    parser.add_argument("--patch_picking_strategy", type=str, default="across_layers", help="Strategy for picking patches. Options: 'across_layers', 'last_layer'")
    parser.add_argument("--topk", type=int, default=4, help="Number of top patches to pass to LLM")
    parser.add_argument("--cropping_method", type=str, default="dynamic", help="Method for cropping images. Options: 'dynamic', 'naive'")
    parser.add_argument("--keyword_criteria", type=str, default="description", help="Criteria for selecting keywords. Options: 'description', 'template', 'word")
    parser.add_argument("--positional_information", type=str, default="explicit", help="Method for providing positional information. Options: 'explicit', '2d_before', '2d_after'")
    parser.add_argument("--temperature", type=float, default=0.2)

    args = parser.parse_args()
    inference(args)