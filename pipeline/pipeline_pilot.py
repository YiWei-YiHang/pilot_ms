import mindspore as ms
import PIL
import numpy as np

from mindformers import CLIPImageProcessor, CLIPTokenizer
from mindone.transformers import CLIPTextModel, CLIPVisionModelWithProjection
from mindone.diffusers.configuration_utils import FrozenDict
from mindone.diffusers.models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, ControlNetModel, ImageProjection, MultiAdapter, T2IAdapter
from mindone.diffusers.schedulers import KarrasDiffusionSchedulers
from mindone.diffusers.utils import (
    #TODO USE_PEFT_BACKEND,
    deprecate,
    #TODO is_accelerate_available,
    #TODO is_accelerate_version,
    logging,
    #TODO replace_example_docstring,
    PIL_INTERPOLATION,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from mindone.diffusers import DiffusionPipeline, StableDiffusionMixin

from mindone.diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from mindone.diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from models.attn_processor import revise_pilot_unet_attention_forward

logger = logging.get_logger(__name__)

class PilotPipeline():
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]