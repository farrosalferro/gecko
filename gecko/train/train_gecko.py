from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
import torch
import os
import wandb
import regex as re
from train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, find_all_linear_names
from conversation import conv_mllava_v1 as default_conv, conv_templates
from gecko.train.data import load_data, load_data_from_config
from pathlib import Path
from typing import Optional
from pathlib import Path

os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
IGNORE_INDEX = -100
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)


@dataclass
class DataArguments:
    train_data_file: Optional[str] = field(
        metadata={"help": "The input training data file (a text file).", "default": None, "required": False},
        default=None,
    )
    val_data_file: Optional[str] = field(
        metadata={"help": "An optional input validation data file (a text file).", "default": None, "required": False},
        default=None,
    )
    test_data_file: Optional[str] = field(
        metadata={"help": "An optional input test data file (a text file).", "default": None, "required": False},
        default=None,
    )
    data_format: Optional[str] = field(
        metadata={"help": "The format of the data file", "default": "chat", "choices": ["chat", "vqa"]},
        default="chat",
    )
    max_seq_len: Optional[int] = field(
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
                          "than this will be truncated.", "default": 1024, "required": False},
        default=1024,
    )
    data_config_file: Optional[str] = field(
        metadata={"help": "Pretrained config name or path if not the same as model_name", "default": None, "required": False},
        default=None,
    )
    dataset_balancing: Optional[bool] = field(
        metadata={"help": "Whether to balance the dataset", "default": True, "required": False},
        default=True,
    )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models",
            "default": "llava-hf/llava-1.5-7b-hf",
            "required": False
        },
        default="llava-hf/llava-1.5-7b-hf",
    )
    lora_enabled: Optional[bool] = field(
        metadata={
            "help": "Whether to use LoRA",
            "default": False,
            "required": False
        },
        default=False,
    )
    qlora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use QLoRA", 
                  "default": False, 
                  "required": False},
        default=False,
    )
    dora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use Dora", 
                  "default": False, 
                  "required": False},
        default=True,
    )
    lora_r: Optional[int] = field(
        metadata={
            "help": "LoRA r",
            "default": 128,
            "required": False
        },
        default=128,
    )
    lora_alpha: Optional[float] = field(
        metadata={
            "help": "LoRA alpha",
            "default": 256,
            "required": False
        },
        default=256,
    )
    lora_dropout: Optional[float] = field(
        metadata={
            "help": "LoRA dropout",
            "default": 0.05,
            "required": False
        },
        default=0.05,
    )
    lora_bias: Optional[str] = field(
        metadata={
            "help": "LoRA bias",
            "default": 'none',
            "required": False
        },
        default='none',
    )
    attn_implementation: Optional[str] = field(
        metadata={
            "help": "The attention implementation to use",
            "default": "flash_attention_2",
            "required": False
        },
        default="flash_attention_2",
    )
    do_pretrain: Optional[bool] = field(
        metadata={
            "help": "Whether to pretrain the projector",
            "default": False,
            "required": False
        },
        default=False,
    )
    llm_backbone: Optional[str] = field(
        metadata={
            "help": "The LLM backbone to use",
            "default": "meta-llama/Meta-Llama-3-8B",
            "required": False
        },
        default="meta-llama/Meta-Llama-3-8B",
    )
    vision_backbone: Optional[str] = field(
        metadata={
            "help": "The vision backbone to use",
            "default": "openai/clip-vit-large-patch14-336",
            "required": False
        },
        default="openai/clip-vit-large-patch14-336",
    )
    conv_template: Optional[str] = field(
        metadata={
            "help": "The conversation template to use",
            "default": None,
            "required": False
        },
        default=None,
    )
    projector: Optional[str] = field(
        metadata={
            "help": "The projector from vision to LLM",
            "default": "MLP",
            "required": False
        },
        default="MLP",
    )
    projector_hidden_act: Optional[str] = field(
        metadata={
            "help": "The hidden activation function of the projector",
            "default": "gelu",
            "required": False
        },
        default="gelu",
    )
    pos_embed_act: Optional[str] = field(
        metadata={
            "help": "The activation function of the positional embedding",
            "default": "gelu",
            "required": False
        },
        default="gelu",
    )
    pe_strategy: Optional[str] = field(
        metadata={
            "help": "The positional embedding strategy",
            "default": "size_coor",
            "required": False
        },
        default="size_coor",
    )
    vision_feature_layer: Optional[int] = field(
        metadata={
            "help": "The vision feature layer",
            "default": -2,
            "required": False
        },
        default=-2,
    )
    num_patches: Optional[int] = field(
        metadata={
            "help": "The number of the selected_patches",
            "default": 1,
            "required": False
        },
        default=1,
    )
    maximum_keyword_tokens: Optional[int] = field(
        metadata={
            "help": "The maximum number of keyword tokens generated by language model",
            "default": 10,
            "required": False
        },
        default=10,
    )
    crop_size: Optional[int] = field(
        metadata={
            "help": "The crop size of the patches. It should follow the input size of the vision backbone",
            "default": 336,
            "required": False
        },
        default=336,
    )
    use_keyword: Optional[bool] = field(
        metadata={
            "help": "Whether to use keyword",
            "default": True,
            "required": False
        },
        default=True,
    )

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_tower', 'vision_resampler', 'pos_embed']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model(model_args, training_args):
    print("Loading model....")
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32

    if model_args.qlora_enabled:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["vision_tower"],
        )
    else:
        bnb_config = None


    from gecko.model import GeckoForConditionalGeneration, GeckoConfig, GeckoProcessor
    if model_args.do_pretrain:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor
        vision_config = AutoConfig.from_pretrained(model_args.vision_backbone).vision_config
        # image_processor = CropImageProcessor(model_args.vision_backbone, model_args.crop_size)
        image_processor = AutoProcessor.from_pretrained(model_args.vision_backbone).image_processor
        vision_backbone = AutoModel.from_pretrained(model_args.vision_backbone, config=vision_config, torch_dtype=torch_dtype, device_map={"": training_args.device})
        
        text_config = AutoConfig.from_pretrained(model_args.llm_backbone)
        llm_backbone = AutoModelForCausalLM.from_pretrained(model_args.llm_backbone, torch_dtype=torch_dtype, device_map={"": training_args.device})
        tokenizer = AutoTokenizer.from_pretrained(model_args.llm_backbone)
        processor = GeckoProcessor(image_processor, tokenizer, model_args.use_keyword, model_args.crop_size, model_args.pe_strategy)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<|pad|>"]})
        
        print("Loading model....")
        config = GeckoConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
            pad_token_id=tokenizer.convert_tokens_to_ids("<|pad|>"),
            vocab_size=len(tokenizer),
            attn_implementaion=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            projector_hidden_act=model_args.projector_hidden_act,
            pos_embed_act=model_args.pos_embed_act,
            pe_strategy=model_args.pe_strategy,
            vision_feature_layer=model_args.vision_feature_layer,
            num_patches=model_args.num_patches,
            multimodal_projector=model_args.projector,
            maximum_keyword_tokens=model_args.maximum_keyword_tokens,
        )
        GeckoForConditionalGeneration._set_default_torch_dtype(torch_dtype)
        model = GeckoForConditionalGeneration(config, vision_backbone, llm_backbone)
    
        # resize token embeddings
        model.language_model.resize_token_embeddings(len(processor.tokenizer))
        model.config.text_config.vocab_size = len(processor.tokenizer)

        for name, param in model.named_parameters():
            if any([x in name for x in ["multi_modal_projector", "pos_embed"]]):
                print("Tuning", name)
            else:
                param.requires_grad = False
        
        print("Successfully loaded model from:", model_args.llm_backbone, model_args.vision_backbone)

    else:
        processor = GeckoProcessor.from_pretrained(model_args.model_name_or_path)
        model = GeckoForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch_dtype,
            attn_implementation=model_args.attn_implementation,
            quantization_config=bnb_config if model_args.qlora_enabled else None
        )
        print("Successfully loaded model from:", model_args.model_name_or_path)

    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False

    if bnb_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if model_args.lora_enabled:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
            use_dora=model_args.dora_enabled,
        )
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
        print("Adding LoRA adapters...")
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        print("Successfully added LoRA adapters")

    return model, processor

def main(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments
):
    if model_args.do_pretrain:
        training_args.output_dir = Path(training_args.output_dir) / model_args.llm_backbone.split("/")[-1] / training_args.run_name
    else:
        training_args.output_dir = Path(training_args.output_dir) / model_args.model_name_or_path.split("/")[-1] / training_args.run_name

    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args.output_dir = str(training_args.output_dir)
    training_args.remove_unused_columns = False
    data_args.is_master_worker = training_args.local_rank in [-1, 0]

    if not training_args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = True
    if training_args.resume_from_checkpoint == True:
        all_checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoints found, training from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint: ", latest_checkpoint)
    
    model, processor = load_model(model_args, training_args)

    if model_args.conv_template:
        data_args.conv_format = conv_templates[model_args.conv_template]
    else:
        if "llama-3" in model.language_model.name_or_path.lower():
            data_args.conv_format = conv_templates['llama_3']
        elif "phi-3" in model.language_model.name_or_path.lower():
            data_args.conv_format = conv_templates['phi_3']
        else:
            data_args.conv_format = default_conv
    print("Using conversation template:", data_args.conv_format)
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data_from_config(data_args, processor)
    else:
        train_dataset, val_dataset, test_dataset, collate_fn = load_data(data_args, processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor
    )
    if trainer.is_world_process_zero():
        print("Training arguments:")
        print(training_args)
        print("Data arguments:")
        print(data_args)
        print("Model arguments:")
        print(model_args)
    if training_args.do_train:
        print("Training model...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # save
        final_checkpoint_dir = os.path.join(training_args.output_dir, 'checkpoint-final')
        if model_args.lora_enabled:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), model_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(final_checkpoint_dir)
                model.save_pretrained(final_checkpoint_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(final_checkpoint_dir, 'non_lora_trainables.bin'))
        else:
            trainer.save_model(output_dir=final_checkpoint_dir)
        processor.save_pretrained(final_checkpoint_dir)
    if training_args.do_predict:
        print("Predicting...")
        trainer.predict(test_dataset)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    main(training_args, data_args, model_args)