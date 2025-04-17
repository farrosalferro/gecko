
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
from copy import deepcopy

import re
import math
import torch
import torch.utils.checkpoint
from torch import nn
import matplotlib.pyplot as plt

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM

from .configuration_gecko import GeckoConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GeckoConfig"

@dataclass
class GeckoCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
class GeckoPreTrainedModel(PreTrainedModel):
    config_class = GeckoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GeckoVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.intializer_range if hasattr(self.config, "intializer_range") else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa

class PositionalEncoding2D(nn.Module):
    def __init__(self, config: GeckoConfig):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        if config.positional_information == "2d_before":
            channels = config.vision_config.hidden_size
        else:
            channels = config.text_config.hidden_size
        self.org_channels = channels
        channels = int(math.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (x, y, num_tokens, ch)
        :return: Positional Encoding Matrix of size (x, y, num_tokens, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        x, y, num_tokens, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1)
        emb_y = self.get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[:, :, None, :orig_ch].repeat(1, 1, num_tokens, 1)
        return self.cached_penc

class GeckoMultiModalProjector(nn.Module):
    def __init__(self, config: GeckoConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
    
class GeckoForConditionalGeneration(GeckoPreTrainedModel):
    def __init__(self, config: GeckoConfig, vision_tower=None, language_model=None, multimodal_projector=None):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config) if vision_tower is None else vision_tower
        self.positional_encoding = PositionalEncoding2D(config) if '2d' in config.positional_information else None
        self.multi_modal_projector = GeckoMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        ) if language_model is None else language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def load_text_encoder(self, processor):
        self.tokenizer = processor.tokenizer
        self.clip_tokenizer = processor.clip_tokenizer
        self.eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        self.encoder_type = self.config.vision_config.model_type
        if 'clip' in self.encoder_type:
            self.encoder = AutoModel.from_pretrained('openai/clip-vit-large-patch14-336', torch_dtype=self.dtype, device_map=self.device)
        elif 'siglip' in self.encoder_type:
            self.encoder = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=self.dtype, device_map=self.device)
        else:
            raise ValueError(f"Vision model {self.config.vision_config.model_type} is not supported.")

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def generate_keywords(self, keywords_text, criteria='template'):
        keywords_text = keywords_text.lstrip('\n')
        first_sentence = keywords_text.split('.')[0] + '.'
        if re.search(r'are (.+?)\.', first_sentence):
            objects = re.search(r'are (.+?)\.', first_sentence).group(1).split(' and ')
        elif re.search(r'is (.+?)\.', first_sentence):
            objects = [re.search(r'is (.+?)\.', first_sentence).group(1)]
        else:
            objects = []
        
        def generate_template(object, description):
            if object[0] in ['a', 'e', 'i', 'o', 'u']:
                return f'An {object}, which {description}'
            else:
                return f'A {object}, which {description}'

        descriptions = []
        keywords = []
        for i, obj in enumerate(objects):
            keywords.append(obj)
            if criteria == 'word':
                descriptions.append([obj])
            elif criteria == 'template':
                descriptions.append([f'a photo of {obj}'])
            elif criteria == 'description':
                # pattern = rf"'{obj}':(.*?)('|\Z)"
                # match = re.search(pattern, keywords_text, re.DOTALL)
                # if match:
                #     # Extract the feature keywords_text and clean it up
                #     feature_text = match.group(1).strip()
                #     # Split on new lines and strip each line
                #     feature_list = [generate_template(obj, line.strip('* ').strip()) for line in feature_text.split('\n') if line.strip()]
                #     descriptions.append(feature_list)
                # The problem of the above code is that it does not work for the case where the object is not found in the text
                # make it more general
                features = re.findall(r"\* (.+)", keywords_text, re.MULTILINE)
                descriptions.append([generate_template(obj, feature) for feature in features[i * len(features) // len(objects): (i + 1) * len(features) // len(objects)]])

            else:
                raise ValueError(f'invalid criteria: {criteria}')
        
        return keywords, descriptions

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images = len(image_features)
        num_image_tokens = torch.tensor([x.shape[0] for x in image_features], device=self.vision_tower.device, dtype=torch.int64) # total image tokens
        embed_dim = image_features[0].shape[-1]
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        # num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        # max_embed_dim = (num_special_image_tokens.max() * (num_image_tokens - 1)) + sequence_length
        max_embed_dim = torch.sum(num_image_tokens) - num_images + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)
        _, image_indices = torch.where(input_ids == self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        image_token_mask = special_image_token_mask * 1
        image_token_mask[0, image_indices] = num_image_tokens - 1
        # for i, index in enumerate(image_indices):
        #     special_image_token_mask[0, index] = num_image_tokens[i] - 1
        new_token_positions = torch.cumsum((image_token_mask) + 1, -1) - 1
        # new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != torch.sum(num_image_tokens):
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = torch.cat([image_patches for image_patches in image_features], dim=0).to(target_device)
        # final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: List[torch.FloatTensor] = None,
        coords: List[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        keyword_prompt_input_ids: torch.LongTensor = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_feature_layer: Optional[int] = None,
        patch_picking_strategy: Optional[str] = None,
        topk: Optional[int] = None,
        keyword_criteria: Optional[str] = None,
        positional_information: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        visualize_patches: Optional[bool] = None,
        visualize_topk_patches: Optional[bool] = None,
        print_keyword: Optional[bool] = None,
    ) -> Union[Tuple, GeckoCausalLMOutputWithPast]:
        """
        Parameters:
            text_inputs: Dict
                Output of tokenizer for text data. A dictionary containing the following keys:
                    - input_ids: torch.LongTensor of shape (batch_size, sequence_length)
                    - attention_mask: torch.LongTensor of shape (batch_size, sequence_length)
                    - token_type_ids: torch.LongTensor of shape (batch_size, sequence_length)
            keyword_inputs: Dict
                Output of tokenizer for keyword data. A dictionary containing the following keys:
                    - input_ids: torch.LongTensor of shape (batch_size, sequence_length)
                    - attention_mask: torch.LongTensor of shape (batch_size, sequence_length)
                    - token_type_ids: torch.LongTensor of shape (batch_size, sequence_length)
            image_inputs: Dict
                Output of ImageProcessor for image data. A dictionary containing the following keys:
                    - pixel_values: torch.FloatTensor of shape (num_images, num_patches, num_tokens, embed_dim)
                    - coords: List of shape (batch_size, num_images)
        """
        # processing image and text inputs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy if vision_feature_select_strategy is not None else self.config.vision_feature_select_strategy
        )
        patch_picking_strategy = patch_picking_strategy if patch_picking_strategy is not None else self.config.patch_picking_strategy
        topk = topk if topk is not None else self.config.topk
        keyword_criteria = keyword_criteria if keyword_criteria is not None else self.config.keyword_criteria
        positional_information = positional_information if positional_information is not None else self.config.positional_information
        visualize_patches = visualize_patches if visualize_patches is not None else self.config.visualize_patches
        visualize_topk_patches = visualize_topk_patches if visualize_topk_patches is not None else self.config.visualize_topk_patches
        print_keyword = print_keyword if print_keyword is not None else self.config.print_keyword

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:

                pixel_values = [pixel_value.to(self.vision_tower.device, dtype=self.vision_tower.dtype) for pixel_value in pixel_values]
                if topk < 0:
                    raise ValueError(f"topk should be greater than or equal to 0, got {topk}")

                elif topk > 0:
                    with torch.no_grad():
                        keyword_input_ids = self.language_model.generate(keyword_prompt_input_ids, return_dict_in_generate=True, max_new_tokens=1024, eos_token_id=self.eos_token_id)
                        keyword_input_ids = keyword_input_ids.sequences[:, keyword_prompt_input_ids.shape[-1]:]
                    keyword_text = self.tokenizer.decode(keyword_input_ids[0], skip_special_tokens=True)

                    # print(keyword_text)
                    generated_keywords, generated_descriptions = self.generate_keywords(keyword_text, criteria=keyword_criteria)

                    all_text_features = []
                    for descriptions in generated_descriptions:
                        one_text_features = []
                        for description in descriptions:
                            keyword_ids = self.clip_tokenizer(description, return_tensors='pt')
                            keyword_ids = {k: v.to(self.device) for k, v in keyword_ids.items()}
                            text_features = self.encoder.get_text_features(**keyword_ids)
                            one_text_features.append(text_features / text_features.norm(p=2, dim=-1, keepdim=True))
                        all_text_features.append(torch.cat(one_text_features, dim=0))

                    selected_image_features = []
                    selected_coords = []
                    for p, pixel_value in enumerate(pixel_values): # iterate through each image
                        print_keyword_text = f'Keywords (criteria: {keyword_criteria}):\n'
                        batching_size_pixel_values = 3
                        all_hidden_states = self.vision_tower(pixel_value[:batching_size_pixel_values], output_hidden_states=True).hidden_states # tuple of size (num_layers, batch_num_patch, num_tokens, vison_embed_dim)
                        for i in range(batching_size_pixel_values, len(pixel_value), batching_size_pixel_values):
                            pixel_value_batch = pixel_value[i:min(i+batching_size_pixel_values, len(pixel_value))]
                            batch_hidden_states = self.vision_tower(pixel_value_batch, output_hidden_states=True).hidden_states # tuple of size (num_layers, batch_num_patch, num_tokens, vison_embed_dim)
                            all_hidden_states = [torch.cat([all_hidden_states[j], batch_hidden_states[j]], dim=0) for j in range(len(all_hidden_states))]

                        # all_hidden_states = self.vision_tower(pixel_value, output_hidden_states=True).hidden_states # tuple of size (num_layers, num_patch, num_tokens, vison_embed_dim)
                        if patch_picking_strategy == 'last_layer':
                            hidden_states = [all_hidden_states[-1]]
                        elif patch_picking_strategy == 'across_layers':
                            hidden_states = deepcopy(all_hidden_states)
                        top_patches = [0]
                        for i, text_feature in enumerate(all_text_features):
                            print_keyword_text += f'  {i+1}: ' + "\n     ".join(generated_descriptions[i]) + '\n'
                            top_index = []
                            for hidden_state in hidden_states: # iterate through each layer
                                if 'clip' in self.encoder_type:
                                    if vision_feature_select_strategy == 'cls':
                                        image_features = self.encoder.visual_projection(self.encoder.vision_model.post_layernorm(hidden_state[1:, 0, :])) # (num_patch-1, embed_dim)
                                    elif vision_feature_select_strategy == 'image_features':
                                        image_features = self.encoder.visual_projection(self.encoder.vision_model.post_layernorm(hidden_state[1:, 1:, :])) # (num_patch-1 * num_tokens, embed_dim)
                                    num_tokens = hidden_state.shape[1] - 1
                                elif 'siglip' in self.encoder_type:
                                    if vision_feature_select_strategy == 'cls':
                                        image_features = self.encoder.vision_model.head(self.encoder.vision_model.post_layernorm(hidden_state[1:, :, :])) # (num_patch-1, embed_dim)
                                    elif vision_feature_select_strategy == 'image_features':
                                        image_features = self.encoder.vision_model.post_layernorm(hidden_state[1:, :, :]) # (num_patch-1 * num_tokens, embed_dim)
                                    num_tokens = hidden_state.shape[1]
                                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                                if vision_feature_select_strategy == 'cls':
                                    similarity_score = torch.matmul(image_features, text_feature.T).mean(dim=1) # (num_patch-1)
                                    if patch_picking_strategy == 'across_layers':
                                        index = torch.topk(similarity_score, 1).indices
                                        top_index.append(index.item()+1)
                                    elif patch_picking_strategy == 'last_layer':
                                        index = torch.topk(similarity_score, math.ceil(topk / len(all_text_features))).indices + 1 # take top k patches
                                        top_index += index.tolist()
                                elif vision_feature_select_strategy == 'image_features':
                                    image_features = image_features.flatten(0, 1)
                                    similarity_score = torch.matmul(image_features, text_feature.T).mean(dim=1) # (num_patch-1 * num_tokens)
                                    index = torch.topk(similarity_score, 100).indices # take top 100 tokens
                                    patch_index = torch.floor(index / num_tokens) # get the patch index
                                    count = torch.nn.functional.one_hot(patch_index.to(torch.int64)).sum(dim=0) # count the occurrences of each patch
                                    if patch_picking_strategy == 'across_layers':
                                        top_count = torch.topk(count, 1).indices # take top 1
                                        top_index.append(top_count.item()+1)
                                    elif patch_picking_strategy == 'last_layer':
                                        top_count = torch.topk(count, math.ceil(topk / len(all_text_features))).indices + 1
                                        top_index += top_count.tolist()

                            if visualize_patches and patch_picking_strategy == 'across_layers':
                                if 'clip' in self.encoder_type:
                                    (x, y) = (5, 5)
                                elif 'siglip' in self.encoder_type:
                                    (x, y) = (7, 4)
                                fig, axs = plt.subplots(y, x, figsize=(x * 2, y * 2))
                                fig.suptitle(f'keyword: {generated_keywords[i]}')
                                for k, index in enumerate(top_index):
                                    axs[k // x, k % x].imshow(pixel_value[index].to(torch.float32).cpu().numpy().transpose(1, 2, 0))
                                    axs[k // x, k % x].set_title(f'Layer {k}')
                                    axs[k // x, k % x].axis('off')
                                plt.show()
                            if patch_picking_strategy == 'across_layers':
                                top_patches += torch.topk(torch.bincount(torch.tensor(top_index, dtype=torch.int64)), math.ceil(topk / len(all_text_features))).indices.to(dtype=torch.int64).tolist()
                            elif patch_picking_strategy == 'last_layer':
                                top_patches += top_index
                        topk_patches = list(set(top_patches))
                        if visualize_topk_patches:
                            fig, axs = plt.subplots(1, len(topk_patches), figsize=(len(topk_patches) * 2, 2))
                            fig.suptitle(f'top-{len(topk_patches)} patches')
                            for m, topk_patch in enumerate(topk_patches):
                                axs[m].imshow(pixel_value[topk_patch].to(torch.float32).cpu().numpy().transpose(1, 2, 0))
                                axs[m].axis('off')
                            plt.show()

                        if 'clip' in self.encoder_type:
                            selected_image_features.append(all_hidden_states[vision_feature_layer][topk_patches, 1:, :])
                        elif 'siglip' in self.encoder_type:
                            selected_image_features.append(all_hidden_states[vision_feature_layer][topk_patches, :, :])
                        selected_coords.append([coords[p][q-1] for q in topk_patches[1:]])
                # if isinstance(pixel_values, list):
                #     pixel_values = torch.cat([x for x in pixel_values if x is not None], dim=0)
                else:
                    selected_image_features = []
                    selected_coords = []
                    for p, pixel_value in enumerate(pixel_values):
                        hidden_states = self.vision_tower(pixel_value, output_hidden_states=True).hidden_states[vision_feature_layer]
                        if 'clip' in self.encoder_type:
                            selected_image_features.append([hidden_states[0, 1:, :]])
                        elif 'siglip' in self.encoder_type:
                            selected_image_features.append([hidden_states[0, :, :]])
                        selected_coords.append([]) # no coordinates
                    print_keyword_text = "No keywords provided."

                if print_keyword:
                    print(print_keyword_text)
                multimodal_projector_features = []
                
                for x, (selected_image_feature, selected_coord) in enumerate(zip(selected_image_features, selected_coords)):
                    print(f'image {x+1}: {selected_coord}')
                    if '2d' in positional_information:
                        max_width = max(selected_coord, key= lambda x: x[0])[0] + 1
                        max_height = max(selected_coord, key= lambda x: x[1])[1] + 1
                        positional_encoding = self.positional_encoding(torch.ones((max_width, max_height, selected_image_feature.shape[1], self.positional_encoding.org_channels), dtype=self.dtype, device=self.device))
                    accumulate = []
                    for i, top_patch in enumerate(selected_image_feature):
                        if positional_information == '2d_before' and i != 0:
                            top_patch += positional_encoding[selected_coord[i-1][0], selected_coord[i-1][1], :, :]
                        aligned_image_feature = self.multi_modal_projector(top_patch)
                        if positional_information == '2d_after' and i != 0:
                            aligned_image_feature += positional_encoding[selected_coord[i-1][0], selected_coord[i-1][1], :, :]
                        accumulate.append(aligned_image_feature)
                        if i == 0:
                            accumulate.append(self.get_input_embeddings()(self.tokenizer(', ', padding=False, truncation=False, max_length=None, return_tensors='pt')['input_ids'].to(device=self.device)[0, 1:]))
                            continue
                        if positional_information == 'explicit':
                            accumulate.append(self.get_input_embeddings()(self.tokenizer(f' at {str(selected_coord[i-1])}, ', padding=False, truncation=False, max_length=None, return_tensors='pt')['input_ids'].to(device=self.device)[0, 1:]))
                        else:
                            accumulate.append(self.get_input_embeddings()(self.tokenizer(f', ', padding=False, truncation=False, max_length=None, return_tensors='pt')['input_ids'].to(device=self.device)[0, 1:]))
                    multimodal_projector_features.append(torch.cat(accumulate, dim=0)) # dimension of (num_selected_patch * num_tokens-1 + num_selected_patch * sep_len - 1) -> (num_selected_patch * num_tokens - 1) as sep_len = 1
                
                assert len(selected_image_features) == len(multimodal_projector_features), f"The number of selected image features and image features do not match. Dimension of selected image features: {len(selected_image_features)} and dimension of image features: {len(multimodal_projector_features)}."
                # print(multimodal_projector_features[0].shape)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    multimodal_projector_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]
        
        batch_shift = 100
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                logits_shape = logits.shape
                labels_shape = labels.shape
                shift_attention_mask_shape = shift_attention_mask.shape
                for i in range(0, shift_attention_mask.shape[1], batch_shift):
                    shift_logits = logits[..., i:min(i+batch_shift, logits_shape[1]-1), :][shift_attention_mask[..., i:min(i+batch_shift, shift_attention_mask_shape[1])].to(logits.device) != 0].contiguous()
                    shift_labels = labels[..., i+1:min(i+batch_shift+1, labels_shape[1])][shift_attention_mask[..., i:min(i+batch_shift, shift_attention_mask_shape[1])].to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return GeckoCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, keyword_prompt_input_ids=None, coords=None, **kwargs
    ):

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "keyword_prompt_input_ids": keyword_prompt_input_ids,
                "coords": coords,
                "topk": kwargs.get("topk"),
                "vision_feature_select_strategy": kwargs.get("vision_feature_select_strategy"),
                "vision_feature_layer": kwargs.get("vision_feature_layer"),
                "patch_picking_strategy": kwargs.get("patch_picking_strategy"),
                "keyword_criteria": kwargs.get("keyword_criteria"),
                "positional_information": kwargs.get("positional_information"),
                "visualize_patches": kwargs.get("visualize_patches"),
                "visualize_topk_patches": kwargs.get("visualize_topk_patches"),
                "print_keyword": kwargs.get("print_keyword"),
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
