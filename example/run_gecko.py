from gecko.model import GeckoForConditionalGeneration, GeckoConfig, GeckoProcessor, chat_gecko
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor
from PIL import Image
import torch

image1 = "0004.png"
image2 = "example/image1.jpg"
# images = [Image.open(image2)]
vision_model = "openai/clip-vit-large-patch14-336"
llm_model = "meta-llama/Meta-Llama-3-8B-Instruct"
torch_dtype = torch.float16

vision_config = AutoConfig.from_pretrained(vision_model).vision_config
image_processor = AutoProcessor.from_pretrained(vision_model).image_processor
vision_backbone = AutoModel.from_pretrained(vision_model, torch_dtype=torch_dtype, config=vision_config, device_map='cuda')
text_config = AutoConfig.from_pretrained(llm_model, trust_remote_code=True)
llm_backbone = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch_dtype, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<|pad|>"]})
processor = GeckoProcessor(image_processor, tokenizer, use_keyword=False, crop_size=336)

config = GeckoConfig(
    vision_config=vision_config, 
    text_config=text_config, 
    attn_implementation='flash_attention_2',
    image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<|pad|>"),
    vocab_size=len(tokenizer),
    use_keyword=True
    )

GeckoForConditionalGeneration._set_default_torch_dtype(torch_dtype)
model = GeckoForConditionalGeneration(
    config=config, 
    vision_tower=vision_backbone, 
    language_model=llm_backbone)
model.language_model.resize_token_embeddings(len(tokenizer))
model.config.text_config.vocab_size = len(tokenizer)

generation_kwargs = {
    "max_new_tokens": 1024,
    "num_beams": 1,
    "do_sample": False
}

# chat
images=None
text = "nWhat animal is this picture?"
response, history = chat_gecko(text, images, model, processor, **generation_kwargs)

print("USER: ", text)
print("ASSISTANT: ", response)

