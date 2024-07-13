"""
Creating different components of the model from config.
"""

import typing as tp
from typing import Any, Dict

import numpy as np

from conditioners import MultiConditioner, NumberConditioner, T5Conditioner
from model import ConditionedDiffusionModelWrapper, DiTWrapper
from pretransforms import AudioAutoencoder, AutoencoderPretransform, OobleckDecoder

"""
Conditioner
"""


def create_multi_conditioner_from_conditioning_config(
    config: tp.Dict[str, tp.Any]
) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]

    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)


"""
Autoencoder/Pretransform
"""


def create_autoencoder_from_config(config: Dict[str, Any]):

    ae_config = config["model"]

    decoder = create_decoder_from_config(ae_config["decoder"])

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert (
        downsampling_ratio is not None
    ), "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    return AudioAutoencoder(
        None,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=None,
        pretransform=None,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=False,
    )


def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(**decoder_config["config"])
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")

    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder


def create_pretransform_from_config(pretransform_config, sample_rate):
    pretransform_type = pretransform_config.get("type", None)

    assert (
        pretransform_type is not None
    ), "type must be specified in pretransform config"

    if pretransform_type == "autoencoder":
        # Create fake top-level config to pass sample rate to autoencoder constructor
        # This is a bit of a hack but it keeps us from re-defining the sample rate in the config
        autoencoder_config = {
            "sample_rate": sample_rate,
            "model": pretransform_config["config"],
        }
        autoencoder = create_autoencoder_from_config(autoencoder_config)

        scale = pretransform_config.get("scale", 1.0)
        model_half = pretransform_config.get("model_half", False)
        iterate_batch = pretransform_config.get("iterate_batch", False)

        pretransform = AutoencoderPretransform(
            autoencoder,
            scale=scale,
            model_half=model_half,
            iterate_batch=iterate_batch,
            chunked=False,
        )
    else:
        raise NotImplementedError(f"Unknown pretransform type: {pretransform_type}")

    enable_grad = pretransform_config.get("enable_grad", False)
    pretransform.enable_grad = enable_grad

    pretransform.eval().requires_grad_(pretransform.enable_grad)

    return pretransform


"""
Diffusion
"""


def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):

    model_config = config["model"]

    model_type = config["model_type"]

    diffusion_config = model_config.get("diffusion", None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_model_type = diffusion_config.get("type", None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"

    diffusion_model_config = diffusion_config.get("config", None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    if diffusion_model_type == "dit":
        diffusion_model = DiTWrapper(**diffusion_model_config)

    io_channels = model_config.get("io_channels", None)
    assert io_channels is not None, "Must specify io_channels in model config"

    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "Must specify sample_rate in config"

    diffusion_objective = diffusion_config.get("diffusion_objective", "v")

    conditioning_config = model_config.get("conditioning", None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(
            conditioning_config
        )

    cross_attention_ids = diffusion_config.get("cross_attention_cond_ids", [])
    global_cond_ids = diffusion_config.get("global_cond_ids", [])
    input_concat_ids = diffusion_config.get("input_concat_ids", [])
    prepend_cond_ids = diffusion_config.get("prepend_cond_ids", [])

    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if diffusion_model_type == "adp_cfg_1d" or diffusion_model_type == "adp_1d":
        min_input_length *= np.prod(diffusion_model_config["factors"])
    elif diffusion_model_type == "dit":
        min_input_length *= diffusion_model.model.patch_size

    # Get the proper wrapper class

    extra_kwargs = {}

    if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint":
        wrapper_fn = ConditionedDiffusionModelWrapper

        extra_kwargs["diffusion_objective"] = diffusion_objective

    return wrapper_fn(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        **extra_kwargs,
    )
