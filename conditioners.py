"""
Conditioners
see https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/conditioners.py
"""

# Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py
import logging
import typing as tp
import warnings
from math import pi
from typing import List, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


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

                # Unwrap the condition info if it's a single-element list or tuple
                # this is to support collation functions that wrap everything in a list
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
