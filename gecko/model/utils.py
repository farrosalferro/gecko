import PIL
import torch
from .modelling_gecko import GeckoForConditionalGeneration
from .processing_gecko import GeckoProcessor
from .conversation import conv_llama_3 as default_conv, conv_templates
import transformers

from typing import List, Tuple, Union

def chat_gecko(
    text:str, 
    images: List[Union[PIL.Image.Image, str]], 
    model:GeckoForConditionalGeneration, 
    processor:GeckoProcessor, 
    max_input_length:int=None, 
    history:List[dict]=None, 
    **kwargs) -> Tuple[str, List[dict]]:

    if "llama-3" in model.language_model.name_or_path.lower():
        conv = conv_templates['llama_3']
        terminators = [
            processor.tokenizer.eos_token_id,
            processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    elif "phi-3" in model.language_model.name_or_path.lower():
        conv = conv_templates['phi_3']
        terminators = [
            processor.tokenizer.eos_token_id,
            processor.tokenizer.convert_tokens_to_ids("<|end|>")
        ]
    else:
        conv = default_conv
        terminators = None

    kwargs["eos_token_id"] = terminators
    conv = conv.copy()
    conv.messages = []
    if history is not None:
        for message in history:
            assert message["role"] in conv.roles
            conv.append_message(message["role"], message["text"])
        if text:
            assert conv.messages[-1][0] == conv.roles[1], "The last message in the history should be the assistant, if the given text is not empty"
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], "")
            history.append({"role": conv.roles[0], "text": text})
            history.append({"role": conv.roles[1], "text": ""})
        else:
            if conv.messages[-1][0] == conv.roles[1]:
                assert conv.messages[-1][1] == "", "No user message should be provided"
            else:
                assert conv.messages[-1][0] == conv.roles[0], "The last message in the history should be the user, if the given text is empty"
                conv.append_message(conv.roles[0], "")
                history.append({"role": conv.roles[0], "text": ""})
    else:
        history = []
        history.append({"role": conv.roles[0], "text": text})
        history.append({"role": conv.roles[1], "text": ""})
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], "")
    assert conv.messages[-1][0] == conv.roles[1] and conv.messages[-1][1] == "", "Format check"
    assert history[-1]["role"] == conv.roles[1] and history[-1]["text"] == "", "Format check"
    
    keyword_prompt = conv.get_keyword_prompt()
    prompt = conv.get_prompt()
    if images:
        for i in range(len(images)):
            if isinstance(images[i], str):
                images[i] = PIL.Image.open(images[i]).convert("RGB")
    
    inputs = processor(images=images, text=prompt, keywords_text=keyword_prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    for k, v in inputs.items():
        if v is not None:
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
            elif isinstance(v, list):
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
            
    output_ids = model.generate(**inputs, **kwargs)[0]

    # remove the input tokens
    generated_ids = output_ids[inputs["input_ids"].shape[-1]:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    history[-1]["text"] = generated_text

    return generated_text, history
            
