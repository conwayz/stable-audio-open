"""
Pretransform/decoder
see https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/autoencoders.py
and https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/pretransforms.py
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


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
