import math
from typing import List, Optional, Union, Dict

import torch
from PIL import Image
import logging

import os
import json
import re
from transformers.feature_extraction_sequence_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers import ProcessorMixin, ImageProcessingMixin, AutoImageProcessor, AutoTokenizer, AutoProcessor
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from transformers.processing_utils import transformers_module
from transformers.utils.hub import is_remote_url, download_url, cached_file, is_offline_mode
from transformers.utils import IMAGE_PROCESSOR_NAME

logger = logging.getLogger(__name__)


class GeckoProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("CLIPImageProcessor", "SiglipImageProcessor")
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, use_keyword=False, crop_size=336, cropping_method='dynamic', **kwargs):
        super().__init__(image_processor, tokenizer)
        self.crop_size = crop_size if crop_size is not None else int(image_processor.size['height'])
        self.use_keyword = use_keyword
        self.image_token_index = None
        self.cropping_method = cropping_method
        self.load_clip_tokenizer()

    def load_clip_tokenizer(self):
        if 'clip' in self.image_processor.image_processor_type.lower():
            self.clip_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')
        elif 'siglip' in self.image_processor.image_processor_type.lower():
            self.clip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
        else:
            raise ValueError(f"Invalid image processor type: {self.image_processor.image_processor_type}")

    def process_images(self, images: List[Image.Image]):
        # create documentation
        """
        Parameters:
            images: List[Image.Image]
                List of PIL images to be processed
        Returns:
            Dict[str, torch.Tensor]:
                pixel_values: List[torch.Tensor]
                    Pixel values of the images. Has shape (num_images, num_patches, num_channels, height, width)
                coords: List[List[List[int]]]
                    Coordinates of the cropped images. Has shape (num_images, num_patches, 2)
                """

        pixel_values = []
        coords = []

        for image in images:
            outputs, coord = self.dynamic_preprocess(image)
            pixel_values.append(outputs)
            coords.append(coord)

        return {"pixel_values": pixel_values, "coords": coords}

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        if self.cropping_method == 'dynamic':
            max_num = math.ceil(orig_width / self.crop_size) * math.ceil(orig_height / self.crop_size)

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= 1)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, self.crop_size)
            # if target_aspect_ratio[0] * target_aspect_ratio[1] <= 25:
            #     target_aspect_ratio = (int(1.5 * target_aspect_ratio[0]), int(1.5 * target_aspect_ratio[1]))
            
        elif self.cropping_method == 'naive':
            target_aspect_ratio = (orig_width // self.crop_size, orig_height // self.crop_size)
            # print(target_aspect_ratio)
            # if target_aspect_ratio[0] * target_aspect_ratio[1] <= 25:
            #     target_aspect_ratio = (2 * orig_width // self.crop_size, 2 * orig_height // self.crop_size)
            # print(target_aspect_ratio)
        else:
            raise ValueError(f"Invalid cropping method: {self.cropping_method}")

        # calculate the target width and height
        target_width = self.crop_size * target_aspect_ratio[0]
        target_height = self.crop_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # add whole image
        processed_images = []
        processed_images.append(image.resize((self.crop_size, self.crop_size)))
        coords = []
        if blocks == 1:
            return self.image_processor(images=processed_images, return_tensors='pt')['pixel_values'], coords

        # resize the image
        resized_img = image.resize((target_width, target_height))
        for i in range(blocks):
            x0 = (i % (target_width // self.crop_size))
            y0 = (i // (target_width // self.crop_size))
            x1 = ((i % (target_width // self.crop_size)) + 1)
            y1 = ((i // (target_width // self.crop_size)) + 1)

            box = (
                x0 * self.crop_size,
                y0 * self.crop_size,
                x1 * self.crop_size,
                y1 * self.crop_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

            coords.append([x0, y0])

            # box = (
            #     (i % (target_width // self.crop_size)) * self.crop_size,
            #     (i // (target_width // self.crop_size)) * self.crop_size,
            #     ((i % (target_width // self.crop_size)) + 1) * self.crop_size,
            #     ((i // (target_width // self.crop_size)) + 1) * self.crop_size
            # )
            # split the image

        assert len(processed_images) == blocks + 1

        return self.image_processor(images=processed_images, return_tensors='pt')['pixel_values'], coords
    

    def preprocess_interleaved_images_and_text(
        self,
        text,
        images=None,
    ):
        """
        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                text can contain <image> tokens as the placeholder for the image(s) to be inserted.
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`, `List[List[PIL.Image.Image]]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
                the number of the images should match the number of <image> tokens in the text.
        
        """
        assert text is not None, "text cannot be None."
            
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            if isinstance(images, list) and isinstance(images[0], Image.Image):
                if isinstance(text, str):
                    images = [images]
                elif isinstance(text, list):
                    if len(text) != len(images):
                        raise ValueError("Invalid input text. Number of texts does not match number of images.")
                    images = [[image] for image in images]
            if isinstance(text, str):
                num_images = len(images[0])    
                num_image_tokens = text.count("<image>")
                if num_image_tokens < num_images:
                    # prepend empty image tokens to text
                    if "USER:" in text:
                        text = text.replace("USER:", "USER:" + "<image>" * (num_images - num_image_tokens), 1)
                    elif "Human:" in text:
                        text = text.replace("Human:", "Human:" + "<image>" * (num_images - num_image_tokens), 1)
                    elif "HUMAN:" in text:
                        text = text.replace("HUMAN:", "HUMAN:" + "<image>" * (num_images - num_image_tokens), 1)
                    else:
                        text = "<image>" * (num_images - num_image_tokens) + text
                    # logger.warning("Image Tokens <image> are not provided in the text. Automatically prepending them before the text. This might cause model to behave unexpectedly.")
                elif num_image_tokens > num_images:
                    text = text.split("<image>")
                    for i, t in enumerate(text):
                        if i < num_images:
                            text[i] = t + "<image>"
                    text = "".join(text)
                    logger.warning(f"Number of <image> tokens: {num_image_tokens} exceeds number of images: {num_images}. Automatically removing extra tokens at the end of the text.")
                    # raise ValueError("Invalid input text. Number of <image> tokens exceeds number of images.")
                texts = [text]
            elif isinstance(text, list):
                if not isinstance(text[0], str):
                    raise ValueError("Invalid input text. Each element of text must be a string.")
                for i, t in enumerate(text):
                    num_image_tokens = t.count("<image>")
                    num_images = len(images[i])
                    if num_image_tokens < num_images:
                        # prepend empty image tokens to text
                        if "USER:" in t:
                            t = t.replace("USER:", "USER:" + "<image>" * (num_images - num_image_tokens), 1)
                        elif "Human:" in t:
                            t = t.replace("Human:", "Human:" + "<image>" * (num_images - num_image_tokens), 1)
                        elif "HUMAN:" in t:
                            t = t.replace("HUMAN:", "HUMAN:" + "<image>" * (num_images - num_image_tokens), 1)
                        else:
                            t = "<image>" * (num_images - num_image_tokens) + t
                        # logger.warning("Image Tokens <image> are not provided in the text. Automatically prepending them before the text. This might cause model to behave unexpectedly.")
                    elif num_image_tokens > num_images:
                        t = t.split("<image>")
                        for j, s in enumerate(t):
                            if j < num_images:
                                t[j] = s + "<image>"
                        t = "".join(t)
                        logger.warning(f"Number of <image> tokens: {num_image_tokens} exceeds number of images: {num_images}. Automatically removing extra tokens at the end of the text.")
                        # raise ValueError("Invalid input text. Number of <image> tokens exceeds number of images.")
                    text[i] = t
                texts = text
            else:
                raise ValueError("Invalid input text. text must be a string or a list of strings.")
            assert all([t.count("<image>") == len(images_per_text) for t, images_per_text in zip(texts, images)]), "Number of <image> tokens in text does not match number of images."
            # add image denotation in text before each <image> as "(image {i}: <image>)"
            for i, t in enumerate(texts):
                for j in range(len(images[i])):
                    t = t.replace("<image>", f"(image {j+1}: <Image><IMAGE></Image>)", 1)
                t = t.replace("<IMAGE>", "<image>")
                texts[i] = t
            
        else:
            if isinstance(text, str):
                texts = [text]
            elif isinstance(text, list):
                if not isinstance(text[0], str):
                    raise ValueError("Invalid input text. Each element of text must be a string.")
                texts = text
            else:
                raise ValueError("Invalid input text. text must be a string or a list of strings.")
        
        return texts, images

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        keywords_text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        add_image_ids: bool = True,
        cropping_method: str = None,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`. Have shape of (num_images, num_patches, num_tokens, embed_dim)
            - **coords** -- Coordinates of the cropped images. Returned when `images` is not `None`. Have shape of (num_images, num_patches, 2)
        """

        if cropping_method is not None:
            self.cropping_method = cropping_method

        if not self.image_token_index:
            self.image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")

        if add_image_ids:
            text, images = self.preprocess_interleaved_images_and_text(text, images)

        text_inputs = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        if self.use_keyword and keywords_text is not None:
            keywords_prompt_input_ids = self.tokenizer(keywords_text,
                                                padding=padding,
                                                truncation=truncation,
                                                max_length=max_length,
                                                return_tensors=return_tensors)['input_ids']
        else:
            keywords_prompt_input_ids = None

        if images is not None:
            input_ids = text_inputs["input_ids"]
            num_image_tokens = torch.sum(input_ids == self.image_token_index, dim=-1)
            for i, num_image_token in enumerate(num_image_tokens):
                if num_image_token < len(images[i]):
                    images[i] = images[i][:num_image_token]
                    print(f"{len(images[i]) - num_image_token} ({len(images[i])} in total) image tokens in the text are truncated due to the max sequence length; removing the extra images.")
            # flatten images
            images = [image for images_per_text in images for image in images_per_text]
            image_inputs = self.process_images(images)
        else:
            image_inputs = {"pixel_values": None, "coords": None}

        return BatchFeature(data={**text_inputs, **image_inputs, "keyword_prompt_input_ids": keywords_prompt_input_ids})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    
    def _right_pad_inputs_with_attention_mask(self, model_inputs: List[Dict]):
        results = {}
        assert len(model_inputs) == 1, "This method only supports a single input, but get {} inputs".format(len(model_inputs))
        for k in model_inputs[0].keys():
            if k == "pixel_values" or k == "coords":
                results[k] = model_inputs[0][k] if model_inputs[0][k] is not None else None
            else:
                results[k] = torch.cat([model_inputs[0][k]], dim=0) if model_inputs[0][k] is not None else None
        return results
    
    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = []
        
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True
            
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, IMAGE_PROCESSOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            processor_file = pretrained_model_name_or_path
            resolved_processor_file = download_url(pretrained_model_name_or_path)
        else:
            processor_file = IMAGE_PROCESSOR_NAME
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=True,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {IMAGE_PROCESSOR_NAME} file"
                )
        
        # Existing processors on the Hub created before #27761 being merged don't have `processor_config.json` (if not
        # updated afterward), and we need to keep `from_pretrained` work. So here it fallbacks to the empty dict.
        # (`cached_file` called using `_raise_exceptions_for_missing_entries=False` to avoid exception)
        # However, for models added in the future, we won't get the expected error if this file is missing.
        if resolved_processor_file is None:
            image_processor_dict = {}

        try:
            # Load processor dict
            with open(resolved_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            image_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
            )
            
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                if attribute_name == "tokenizer":
                    classes = tuple(getattr(transformers_module, n) if n is not None else None for n in class_name)
                    use_fast = kwargs.get("use_fast", True)
                    if use_fast and classes[1] is not None:
                        attribute_class = classes[1]
                    else:
                        attribute_class = classes[0]
                elif attribute_name == "image_processor":
                    image_processor_type = image_processor_dict.get("image_processor_type", None)
                    if image_processor_type is not None:
                        assert image_processor_type in class_name, f"Invalid image processor type: {image_processor_type}"
                        attribute_class = getattr(transformers_module, image_processor_type)
                    else:
                        attribute_class = getattr(transformers_module, class_name[0])
                else:
                    raise ValueError(f"Invalid attribute name: {attribute_name}")
            else:
                attribute_class = getattr(transformers_module, class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args