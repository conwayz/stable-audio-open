import math
from math import pi

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.utils import weight_norm

"""
Conditioners
see https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/conditioners.py
"""

# Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py
import logging
import typing as tp
import warnings
from typing import List, Union


class Conditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int, project_out: bool = False):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = (
            nn.Linear(dim, output_dim)
            if (dim != output_dim or project_out)
            else nn.Identity()
        )

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class NumberConditioner(Conditioner):
    """
    Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    """

    def __init__(self, output_dim: int, min_val: float = 0, max_val: float = 1):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:

        # Cast the inputs to floats
        floats = [float(x) for x in floats]

        floats = torch.tensor(floats).to(device)

        floats = floats.clamp(self.min_val, self.max_val)

        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        float_embeds = self.embedder(normalized_floats).unsqueeze(1)

        return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]


class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-base"]

    T5_MODEL_DIMS = {"t5-base": 768}

    def __init__(
        self,
        output_dim: int,
        t5_model_name: str = "t5-base",
        max_length: str = 128,
        enable_grad: bool = False,
        project_out: bool = False,
    ):
        assert (
            t5_model_name in self.T5_MODELS
        ), f"Unknown T5 model name: {t5_model_name}"
        super().__init__(
            self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out
        )

        from transformers import AutoTokenizer, T5EncoderModel

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = (
                    T5EncoderModel.from_pretrained(t5_model_name)
                    .train(enable_grad)
                    .requires_grad_(enable_grad)
                    .to(torch.float16)
                )
            finally:
                logging.disable(previous_level)

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

    def forward(
        self, texts: tp.List[str], device: tp.Union[torch.device, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()

        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(
            self.enable_grad
        ):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)[
                "last_hidden_state"
            ]

        embeddings = self.proj_out(embeddings.float())

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"}) # noqa: E501
    """

    def __init__(
        self,
        conditioners: tp.Dict[str, Conditioner],
        default_keys: tp.Dict[str, str] = {},
    ):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(
        self,
        batch_metadata: tp.List[tp.Dict[str, tp.Any]],
        device: tp.Union[torch.device, str],
    ) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(
                            f"Conditioner key {condition_key} not found in batch metadata"
                        )

                # Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if (
                    isinstance(x[condition_key], list)
                    or isinstance(x[condition_key], tuple)
                    and len(x[condition_key]) == 1
                ):
                    conditioner_input = x[condition_key][0]

                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            output[key] = conditioner(conditioner_inputs, device)

        return output


class NumberEmbedder(nn.Module):
    def __init__(
        self,
        features: int,
        dim: int = 256,
    ):
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x  # type: ignore


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


"""
Pretransform/decoder
see https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/autoencoders.py
and https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/pretransforms.py
"""


class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def tokenize(self, x):
        raise NotImplementedError

    def decode_tokens(self, tokens):
        raise NotImplementedError


class AutoencoderPretransform(Pretransform):
    def __init__(
        self, model, scale=1.0, model_half=False, iterate_batch=False, chunked=False
    ):
        super().__init__(
            enable_grad=False,
            io_channels=model.io_channels,
            is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete,
        )
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale = scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate

        self.model_half = model_half
        self.iterate_batch = iterate_batch

        self.encoded_channels = model.latent_dim

        self.chunked = chunked

        if self.model_half:
            self.model.half()

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(
            z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs
        )

        if self.model_half:
            decoded = decoded.float()

        return decoded


# oobleck decoder stuff
class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            ),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            ]

        layers += [
            SnakeBeta(c_mults[0] * channels),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Identity(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class SnakeBeta(nn.Module):

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True
    ):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
    ):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )

        self.layers = nn.Sequential(
            SnakeBeta(in_channels),
            upsample_layer,
            *[
                ResidualUnit(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                    use_snake=use_snake,
                )
                for dilation in [1, 3, 9]
            ],
        )

    def forward(self, x):
        return self.layers(x)


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation,
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()

        self.dilation = dilation

        padding = (dilation * (7 - 1)) // 2

        self.layers = nn.Sequential(
            SnakeBeta(out_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
            ),
            SnakeBeta(out_channels),
            WNConv1d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x):
        res = x

        # x = checkpoint(self.layers, x)
        x = self.layers(x)

        return x + res


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck=None,
        pretransform: Pretransform = None,
        in_channels=None,
        out_channels=None,
        soft_clip=False,
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.decoder = decoder

    def decode(self, latents, iterate_batch=False, **kwargs):
        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i : i + 1]))
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **kwargs)

        return decoded

    def decode_audio(
        self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs
    ):
        """
        Decode latents to audio.
        """
        # default behavior. Decode the entire latent in parallel
        return self.decode(latents, **kwargs)


"""
Diffusion
"""
