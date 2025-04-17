import gradio as gr
import spaces
import os
import time
from PIL import Image
import functools
import torch
import matplotlib.pyplot as plt
import re
import ast

from model import GeckoForConditionalGeneration, GeckoConfig, GeckoProcessor, chat_gecko, chat_gecko_stream
from model.conversation import conv_templates
from typing import List

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


# initialization
topk = 1
keyword_criteria = 'word'
positional_information = 'explicit'
vision_feature_select_strategy = 'cls'
patch_picking_strategy = 'last_layer'
cropping_method = 'naive'
crop_size = 384
visualize_topk_patches = False
print_keyword=True
print_topk_patches = True

torch_dtype = torch.float16
attn_implementation = 'sdpa'
device_map = 'cuda'

conv_template = conv_templates['llama_3']

model = 'farrosalferro24/Gecko-Mantis-8B-siglip-llama3'
config = GeckoConfig.from_pretrained(model, 
                                    topk=topk,
                                    visualize_topk_patches=visualize_topk_patches,
                                    keyword_criteria=keyword_criteria,
                                    positional_information=positional_information,
                                    vision_feature_select_strategy=vision_feature_select_strategy,
                                    patch_picking_strategy=patch_picking_strategy,
                                    print_keyword=print_keyword,
                                    print_topk_patches=print_topk_patches)
processor = GeckoProcessor.from_pretrained(model, config=config, use_keyword=True, cropping_method=cropping_method, crop_size=crop_size)
model = GeckoForConditionalGeneration.from_pretrained(
    model, config=config, torch_dtype=torch_dtype, 
    attn_implementation=attn_implementation, device_map=device_map)
model.load_text_encoder(processor)

@spaces.GPU
def generate_stream(text:str, images:List[Image.Image], history: List[dict], **kwargs):
    global processor, model
    model = model.to("cuda")
    if not images:
        images = None
    # print(history)
    print(f'length of images: {len(images)}')
    generator, print_kw, inputs = chat_gecko_stream(text, images, model, processor, history=history, **kwargs)
    texts = []
    # for text, history in chat_gecko_stream(text, images, model, processor, history=history, **kwargs):
    #     yield text
    for text, history in generator:
        texts.append(text)

    # return text
    return texts, print_kw, inputs

@spaces.GPU
def generate(text:str, images:List[Image.Image], history: List[dict], **kwargs):
    global processor, model
    model = model.to("cuda")
    if not images:
        images = None
    generated_text, history = chat_gecko(text, images, model, processor, history=history, **kwargs)
    return generated_text

def enable_next_image(uploaded_images, image):
    uploaded_images.append(image)
    return uploaded_images, gr.MultimodalTextbox(value=None, interactive=False)

def add_message(history, message):
    if message["files"]:
        for file in message["files"]:
            history.append([(file,), None])
    if message["text"]:
        history.append([message["text"], None])
    return history, gr.MultimodalTextbox(value=None)

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def get_chat_history(history):
    chat_history = []
    user_role = conv_template.roles[0]
    assistant_role = conv_template.roles[1]
    for i, message in enumerate(history):
        if isinstance(message[0], str):
            chat_history.append({"role": user_role, "text": message[0]})
            if i != len(history) - 1:
                assert message[1], "The bot message is not provided, internal error"
                chat_history.append({"role": assistant_role, "text": message[1]})
            else:
                assert not message[1], "the bot message internal error, get: {}".format(message[1])
                chat_history.append({"role": assistant_role, "text": ""})
    return chat_history

def get_chat_images(history):
    images = []
    for message in history:
        if isinstance(message[0], tuple):
            images.extend(message[0])
    return images

def bot(history, topk=None, keyword_criteria=None, positional_information=None, vision_feature_select_strategy=None, patch_picking_strategy=None, cropping_method=None):
    print(history)
    cur_messages = {"text": "", "images": []}
    for message in history[::-1]:
        if message[1]:
            break
        if isinstance(message[0], str):
            cur_messages["text"] = message[0] + " " + cur_messages["text"]
        elif isinstance(message[0], tuple):
            cur_messages["images"].extend(message[0])
    cur_messages["text"] = cur_messages["text"].strip()
    cur_messages["images"] = cur_messages["images"][::-1]
    if not cur_messages["text"]:
        raise gr.Error("Please enter a message")
    if cur_messages['text'].count("<image>") < len(cur_messages['images']):
        gr.Warning("The number of images uploaded is more than the number of <image> placeholders in the text. Will automatically prepend <image> to the text.")
        cur_messages['text'] = "<image> "* (len(cur_messages['images']) - cur_messages['text'].count("<image>")) + cur_messages['text']
        history[-1][0] = cur_messages["text"]
    if cur_messages['text'].count("<image>") > len(cur_messages['images']):
        gr.Warning("The number of images uploaded is less than the number of <image> placeholders in the text. Will automatically remove extra <image> placeholders from the text.")
        cur_messages['text'] = cur_messages['text'][::-1].replace("<image>"[::-1], "", cur_messages['text'].count("<image>") - len(cur_messages['images']))[::-1]
        history[-1][0] = cur_messages["text"]
        
    
    
    chat_history = get_chat_history(history)
    chat_images = get_chat_images(history)
    
    generation_kwargs = {
        "max_new_tokens": 4096,
        "num_beams": 1,
        "do_sample": False,
        "topk": topk,
        "keyword_criteria": keyword_criteria,
        "positional_information": positional_information,
        "vision_feature_select_strategy": vision_feature_select_strategy,
        "patch_picking_strategy": patch_picking_strategy,
        "cropping_method": cropping_method,
    }
    
    response = generate_stream(None, chat_images, chat_history, **generation_kwargs)
    num_images = len(response[2].pixel_values)
    coords = response[1][-num_images:]
    print_kw = '\n'.join(response[1][:-num_images-1])
    patches_fig = plot_patches(response[2])
    topk_patches_fig = plot_topk_patches(response[2], coords, topk)
    for _output in response[0]:
        history[-1][1] = _output
        time.sleep(0.05)
        yield history, print_kw, patches_fig, topk_patches_fig

def plot_patches(inputs):
    pixel_value = inputs.pixel_values[0].cpu().numpy()
    x, y = inputs.coords[0][-1][0] + 1, inputs.coords[0][-1][1] + 1

    fig, axes = plt.subplots(y, x, figsize=(x * 4, y * 4))
    for i in range(y):
        for j in range(x):
            axes[i, j].imshow(pixel_value[1 + i * x + j].transpose(1, 2, 0))
            axes[i, j].axis('off')

    return fig

def plot_topk_patches(inputs, selected_coords, topk):
    if topk == 0:
        if len(inputs.pixel_values) == 1:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(inputs.pixel_values[0][0].cpu().numpy().transpose(1, 2, 0))
            plt.axis('off')
            return fig
        else:
            fig, axes = plt.subplots(1, len(inputs.pixel_values), figsize=(len(inputs.pixel_values) * 10, 10))
            for i in range(len(inputs.pixel_values)):
                axes[i].imshow(inputs.pixel_values[i][0].cpu().numpy().transpose(1, 2, 0))
                axes[i].axis('off')
            return fig

    selected_coords_list = []
    for selected_coord in selected_coords:
        match = re.search(r"\[\[.*\]\]", selected_coord)
        if match:
            coordinates_str = match.group(0)
            # Convert the string representation of the list to an actual list
            coordinates = ast.literal_eval(coordinates_str)
            selected_coords_list.append(coordinates)
    num_images = len(selected_coords_list)
    fig, axes = plt.subplots(num_images, len(selected_coords_list[0])+1, figsize=((len(selected_coords_list[0])+1) * 10, num_images * 10))
    if num_images == 1:
        xmax = inputs.coords[0][-1][0] + 1
        for j in range(len(selected_coords_list[0])+1):
            if j == 0:
                axes[j].imshow(inputs.pixel_values[0][0].cpu().numpy().transpose(1, 2, 0))
                axes[j].axis('off')
                continue
            x, y = selected_coords_list[0][j-1][0], selected_coords_list[0][j-1][1]
            axes[j].imshow(inputs.pixel_values[0][1 + y * xmax + x].cpu().numpy().transpose(1, 2, 0))
            axes[j].axis('off')
    else:
        for i in range(num_images):
            xmax = inputs.coords[i][-1][0] + 1
            for j in range(len(selected_coords_list[0])+1):
                if j == 0:
                    axes[i, j].imshow(inputs.pixel_values[i][0].cpu().numpy().transpose(1, 2, 0))
                    continue
                x, y = selected_coords_list[i][j-1][0], selected_coords_list[i][j-1][1]
                axes[i, j].imshow(inputs.pixel_values[i][1 + y * xmax + x].cpu().numpy().transpose(1, 2, 0))
                axes[i, j].axis('off')
    
    return fig


def build_demo():
    with gr.Blocks() as demo:
        
#         gr.Markdown(""" # Mantis
# Mantis is a multimodal conversational AI model that can chat with users about images and text. It's optimized for multi-image reasoning, where inverleaved text and images can be used to generate responses.
# ### [Paper](https://arxiv.org/abs/2405.01483) | [Github](https://github.com/TIGER-AI-Lab/Mantis) | [Models](https://huggingface.co/collections/TIGER-Lab/mantis-6619b0834594c878cdb1d6e4) | [Dataset](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct) | [Website](https://tiger-ai-lab.github.io/Mantis/)            
#         """)
        
#         gr.Markdown("""## Chat with Mantis
#         Mantis supports interleaved text-image input format, where you can simply use the placeholder `<image>` to indicate the position of uploaded images.
#         The model is optimized for multi-image reasoning, while preserving the ability to chat about text and images in a single conversation.
#         (The model currently serving is [🤗 TIGER-Lab/Mantis-8B-siglip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3))
#         """)

        gr.Markdown("""### How to Chat
        To chat, simply enter a message and upload your images. Use the placeholder `<image>` to indicate the position of the uploaded images in the message.
        To do the "identifying small objects" task, set the topk value higher than 0. If you want to do general interleaved text-image chat, set the topk value to 0.
        The model currently serving is the SigLIP variant from [🤗 TIGER-Lab/Mantis-8B-siglip-llama3](https://huggingface.co/TIGER-Lab/Mantis-8B-siglip-llama3)
        """)

        gr.Markdown("""#### Note
        This demo is only for demonstration purpose and not for commercial use. Since we run the model under limited resources, the response time may be slow or even timeout. 
        If your answer is not returned, please try again 5-10 minutes later. Then re-run the demo by clicking the "Demo" button from the previous website. 
        Thank you and sorry for the inconvenience.
        """)
        
        chatbot = gr.Chatbot(line_breaks=True)
        chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload images. Please use <image> to indicate the position of uploaded images", show_label=True)
        
        chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])        

        topk = gr.Slider(
            label='Top-k', 
            minimum=0, 
            maximum=10, 
            step=1, 
            value=1, 
            interactive=True)
        
        with gr.Accordion(label='Advanced options', open=False):
            keyword_criteria = gr.Radio(
                ['word', 'template', 'description'],
                label='Keyword Criteria',
                interactive=True
            )
            positional_information = gr.Radio(
                ['explicit', '2d_before', '2d_after'],
                label='Position Information',
                interactive=True
            )
            vision_feature_select_strategy = gr.Radio(
                ['cls', 'image_features'],
                label='Vision Feature Select Strategy',
                interactive=True
            )
            patch_picking_strategy = gr.Radio(
                ['across_layers', 'last_layer'],
                label='Patch Picking Strategy',
                interactive=True
            )
            cropping_method = gr.Radio(
                ['naive', 'dynamic'],
                label='Cropping Method',
                interactive=True
            )
        
        bot_msg = chat_msg.success(bot, chatbot, 
                                         chatbot, api_name="bot_response")
        
        chatbot.like(print_like_dislike, None, None)

        with gr.Row():
            send_button = gr.Button("Send")
            clear_button = gr.ClearButton([chatbot, chat_input])

        print_kw = gr.Textbox(label="extracted keywords")
        depict_patches = gr.Plot(label="cropped image", format="png")
        depict_topk_patches = gr.Plot(label="top-k image patches", format="png")

        send_button.click(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        ).then(
            bot, 
            [chatbot, topk, keyword_criteria, positional_information, vision_feature_select_strategy, patch_picking_strategy, cropping_method],
            [chatbot, print_kw, depict_patches, depict_topk_patches], api_name="bot_response"
        )
        
        gr.Examples(
            examples=[
                {
                    "text": open("./examples/little_girl.txt").read(),
                    "files": ["./examples/little_girl.jpg"]
                },
                {
                    "text": open("./examples/bus_luggage.txt").read(),
                    "files": ["./examples/bus_luggage.jpg"]
                },
                {
                    "text": open("./examples/child_shoes.txt").read(),
                    "files": ["./examples/child_shoes.png"]
                },
            ],
            inputs=[chat_input],
        )        
        
#         gr.Markdown("""
# ## Citation
# ```
# @article{jiang2024mantis,
#   title={MANTIS: Interleaved Multi-Image Instruction Tuning},
#   author={Jiang, Dongfu and He, Xuan and Zeng, Huaye and Wei, Con and Ku, Max and Liu, Qian and Chen, Wenhu},
#   journal={arXiv preprint arXiv:2405.01483},
#   year={2024}
# }
# ```""")
    return demo    
    

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)
        