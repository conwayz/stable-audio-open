import argparse
import json
import os

import torch
from einops import rearrange
from safetensors.torch import load_file

from factory import create_diffusion_cond_from_config
from generation import generate_diffusion_cond


def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    for key in state_dict:
        if (
            key in model_state_dict
            and state_dict[key].shape == model_state_dict[key].shape
        ):
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]

    model.load_state_dict(model_state_dict, strict=False)


def postprocess(output):
    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")
    # Peak normalize, clip, convert to int16, and save to file
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="demo stable audio open 1.0 generation from prompt"
    )
    parser.add_argument(
        "--model-dir", type=str, help="path to folder containing model weights/config"
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    with open(os.path.join(model_dir, "model_config.json")) as fp:
        model_config = json.load(fp)
    model_ckpt_path = os.path.join(model_dir, "model.safetensors")
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(model_ckpt_path))
    model.to("cuda")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download model
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    # Set up text and timing conditioning
    conditioning = [
        {
            "prompt": "dubstep 128bpm with stabbing synth lead",
            "seconds_start": 0,
            "seconds_total": 47,
        }
    ]

    output = generate_diffusion_cond(
        model,
        conditioning=conditioning,
        sample_size=sample_size,
        device=device,
        seed=42,
    )

    print(output.shape)
    print("done!")
