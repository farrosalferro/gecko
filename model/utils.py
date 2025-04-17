import PIL
import torch
from .modelling_gecko import GeckoForConditionalGeneration
from .processing_gecko import GeckoProcessor
from .conversation import conv_llama_3 as default_conv, conv_templates
import transformers

from typing import List, Tuple, Union
from io import StringIO 
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


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
    
    keyword_prompt = conv.generate_keyword_prompt(text.split("\n")[len(images)])
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
        output_ids = model.generate(**inputs, **kwargs)[0]

    # remove the input tokens
    generated_ids = output_ids[inputs["input_ids"].shape[-1]:]
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)

    history[-1]["text"] = generated_text

    return generated_text, history

def chat_gecko_stream(
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
    
    if images:
        for i in range(len(images)):
            if isinstance(images[i], str):
                images[i] = PIL.Image.open(images[i])
        last_prompt = history[-2]['text'].split("?")[0]
        last_prompt = last_prompt.replace('<image>', '').strip() if '<image>' in last_prompt else last_prompt.strip()
        keyword_prompt = conv.generate_keyword_prompt(last_prompt.replace('<image>', '').strip()) if '<image>' in last_prompt else conv.generate_keyword_prompt(last_prompt.strip())
    else:
        keyword_prompt = None
    prompt = conv.get_prompt()

    cropping_method = kwargs.pop("cropping_method")
    inputs = processor(images=images, text=prompt, keywords_text=keyword_prompt, return_tensors="pt", truncation=True, max_length=max_input_length, cropping_method=cropping_method)
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
    
    from transformers import TextIteratorStreamer
    from threading import Thread
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    kwargs["streamer"] = streamer
    inputs.update(kwargs)
    thread = Thread(target=model.generate, kwargs=inputs)
    thread.start()
    generator = []
    with Capturing() as print_kw:
        for _output in streamer:
            history[-1]["text"] += _output
            generator.append((history[-1]["text"], history))
            # yield history[-1]["text"], history
    return generator, print_kw, inputs
    