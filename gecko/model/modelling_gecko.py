
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.utils.checkpoint
from torch import nn

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

class VisionEncoder(nn.Module):
    def __init__(self, config: GeckoConfig, vision_tower=None):
        super().__init__()
        self.vision_tower = AutoModel.from_config(config.vision_config) if vision_tower is None else vision_tower
        if config.pe_strategy == 'size_coor':
            input_size = 4
        else:
            input_size = 2
        self.pos_embed = nn.Sequential(
            nn.Linear(input_size, config.vision_config.hidden_size) if config.pe_strategy == 'size_coor' else nn.Linear(2, config.vision_config.hidden_size),
            ACT2FN[config.pos_embed_act],
            nn.Linear(config.vision_config.hidden_size, config.vision_config.hidden_size)
        ).to(self.vision_tower.device)

    def process_pixel_values(self, pixel_values, vision_feature_layer):
        for i in range(len(pixel_values)):
            pixel_values[i] = self.vision_tower(pixel_values[i], output_hidden_states=True).hidden_states[vision_feature_layer]
        return pixel_values

    def process_coords(self, coords):
        for i in range(len(coords)):
            coords[i] = self.pos_embed(coords[i]).unsqueeze(1)
        return coords

    def forward(self, pixel_values, coords, vision_feature_layer):
        assert len(pixel_values) == len(coords), f"The number of images from coords and pixel_values do no match. Dimension of coords: {len(coords)} and dimension of pixel_values: {len(pixel_values)}."
        if pixel_values is None:
            return None
        
        pixel_values = self.process_pixel_values(pixel_values, vision_feature_layer)
        coords = self.process_coords(coords)

        for i in range(len(pixel_values)):
            pixel_values[i] += coords[i]

        return pixel_values # shape: (num_images, num_patches, num_tokens, embed_dim)
    
    @property
    def device(self):
        return self.vision_tower.device
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    
class GeckoForConditionalGeneration(GeckoPreTrainedModel):
    def __init__(self, config: GeckoConfig, vision_tower=None, language_model=None, multimodal_projector=None):
        super().__init__(config)
        self.vision_tower = VisionEncoder(config, vision_tower)
        multimodal_projector = config.multimodal_projector if multimodal_projector is None else multimodal_projector
        self.multimodal_projector = multimodal_projector

        if multimodal_projector.lower() == 'mlp':
            from gecko.model.multimodal_encoder import GeckoMLPProjector
            self.multi_modal_projector = GeckoMLPProjector(config).to(self.vision_tower.device)
        elif multimodal_projector.lower() == 'perceiver':
            from gecko.model.multimodal_encoder import GeckoResamplerProjector
            self.multi_modal_projector = GeckoResamplerProjector(config).to(self.vision_tower.device)
        else:
            raise ValueError(f"Multimodal projector {multimodal_projector} is not supported.")
        
        self.vocab_size = config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        ) if language_model is None else language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

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
    
    def _get_highest_similarity(self, cls_token, keyword_hidden_states, top_patches):
        num_patches, embed_dim = cls_token.shape
        batch_size, sequence_length, hidden_size = keyword_hidden_states.shape
        assert embed_dim == hidden_size, f"The embedding dimension of cls token and keyword hidden states do not match. Dimension of cls token: {embed_dim} and dimension of keyword hidden states: {hidden_size}."
        keyword_hidden_states = keyword_hidden_states.squeeze(0)

        # calculate the similarity between the cls token and the keyword hidden states
        similarity_score = torch.matmul(cls_token, keyword_hidden_states.T) # shape: (num_patches, sequence_length)
        similarity_score = similarity_score.mean(dim=1) # shape: (num_patches)
        # take the index of the patch with the highest similarity score
        patch_index = torch.topk(similarity_score, top_patches).indices
        return patch_index
    
    def _select_patches(self, image_features, keyword_hidden_states, top_patches=1):
        selected_patches = []
        # iterate through each image
        for image in image_features:
            if keyword_hidden_states is not None:
                # take the first token of each patch
                cls_token = image[:, 0, :].squeeze(1)
                # get the index of the patch with the highest similarity score
                patch_index = self._get_highest_similarity(cls_token, keyword_hidden_states, top_patches)
            else:
                top_patches = image.shape[0]
                patch_index = torch.arange(top_patches)
            # select the patch with the highest similarity score
            if self.multimodal_projector == 'mlp':
                image = image[patch_index, 1:, :].reshape(-1, image.shape[-1]).type(self.vision_tower.dtype)
            elif self.multimodal_projector == 'perceiver':
                image = image[patch_index, :, :].reshape(-1, image.shape[-1]).type(self.vision_tower.dtype)
            else:
                raise ValueError(f"Multimodal projector {self.multimodal_projector} is not supported.")
            selected_patches.append(image)
        return selected_patches # shape: list with shape of num_images, each element of shape (num_tokens * num_patches_i, embed_dim)
    
    def _input_to_multimodal_projector(self, selected_image_features):
        output = []
        for selected_image in selected_image_features:
            selected_image = self.multi_modal_projector(selected_image)
            output.append(selected_image)
        return output # shape: list with shape of num_images, each element of shape (num_patches_i, num_tokens, embed_dim) where i is the index of the image
    
    def _process_keyword_input(self, keyword_input_ids, maximum_keyword_tokens=10):
        self.language_model.eval()
        with torch.no_grad():
            output_ids = self.language_model.generate(input_ids=keyword_input_ids, return_dict_in_generate=True, max_new_tokens=maximum_keyword_tokens)
            output_ids = output_ids.sequences[:, keyword_input_ids.shape[-1]:]

        self.language_model.train()
        # conditions
        if output_ids[0, 0:2].tolist() == [35581, 25]: # condition where the output is in the form Keyword: <keyword>
            keyword_ids = output_ids[:, 2:-1]
            if keyword_ids[0, 0].item() == 482:
                return None
            return self.get_input_embeddings()(keyword_ids)
        else: # output
            return None

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
        keyword_input_ids: torch.LongTensor = None,
        vision_feature_layer: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                pixel_values = [pixel_value.to(self.vision_tower.device, dtype=self.vision_tower.dtype) for pixel_value in pixel_values]
                coords = [coord.to(self.vision_tower.device, dtype=self.vision_tower.dtype) for coord in coords]
                image_features = self.vision_tower(pixel_values, coords, vision_feature_layer)
                
                if keyword_input_ids is not None:
                    keyword_hidden_states = self._process_keyword_input(keyword_input_ids, self.config.maximum_keyword_tokens)
                else:
                    keyword_hidden_states = None

                # if isinstance(pixel_values, list):
                #     pixel_values = torch.cat([x for x in pixel_values if x is not None], dim=0)
                image_features = self._input_to_multimodal_projector(image_features)
                selected_image_features = self._select_patches(image_features, keyword_hidden_states, self.config.num_patches)
                assert len(selected_image_features) == len(image_features), f"The number of selected image features and image features do not match. Dimension of selected image features: {len(selected_image_features)} and dimension of image features: {len(image_features)}."
                print(selected_image_features[0].shape)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    selected_image_features, inputs_embeds, input_ids, attention_mask, labels
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
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, coords=None, keyword_input_ids=None, **kwargs
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
                "coords": coords,
                "keyword_input_ids": keyword_input_ids,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
