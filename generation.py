import typing as tp

import k_diffusion as K
import numpy as np
import torch


def generate_diffusion_cond(
    model,
    steps: int = 250,
    cfg_scale=6,
    conditioning: dict = None,
    conditioning_tensors: tp.Optional[dict] = None,
    negative_conditioning: dict = None,
    negative_conditioning_tensors: tp.Optional[dict] = None,
    batch_size: int = 1,
    sample_size: int = 2097152,
    sample_rate: int = 48000,
    seed: int = -1,
    device: str = "cuda",
    init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
    init_noise_level: float = 1.0,
    mask_args: dict = None,
    return_latents=False,
    **sampler_kwargs,
) -> torch.Tensor:
    """
    Generate audio from a prompt using a diffusion model.

    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.
    """
    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio

    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert (
        conditioning is not None or conditioning_tensors is not None
    ), "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    negative_conditioning_tensors = {}

    # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch.
    init_audio = None
    mask = None

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {
        k: v.type(model_dtype) if v is not None else v
        for k, v in conditioning_inputs.items()
    }

    # Now the generative AI part:
    # k-diffusion denoising process go!

    diff_objective = model.diffusion_objective

    if diff_objective == "v":
        # k-diffusion denoising process go!
        sampled = sample_k(
            model.model,
            noise,
            init_audio,
            mask,
            steps,
            **sampler_kwargs,
            **conditioning_inputs,
            **negative_conditioning_tensors,
            cfg_scale=cfg_scale,
            batch_cfg=True,
            rescale_cfg=True,
            device=device,
        )
    else:
        raise NotImplementedError

    # v-diffusion:
    # sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done.
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        # cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled


# Uses k-diffusion from https://github.com/crowsonkb/k-diffusion
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, set both init_data and mask to None
# For variations, set init_data
# For inpainting, set both init_data & mask
def sample_k(
    model_fn,
    noise,
    init_data=None,
    mask=None,
    steps=100,
    sampler_type="dpmpp-2m-sde",
    sigma_min=0.5,
    sigma_max=50,
    rho=1.0,
    device="cuda",
    callback=None,
    cond_fn=None,
    **extra_args,
):
    denoiser = K.external.VDenoiser(model_fn)

    # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
    sigmas = K.sampling.get_sigmas_polyexponential(
        steps, sigma_min, sigma_max, rho, device=device
    )
    # Scale the initial noise by sigma
    noise = noise * sigmas[0]

    wrapped_callback = callback

    # SAMPLING
    # set the initial latent to noise
    x = noise

    with torch.cuda.amp.autocast():
        if sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(
                denoiser,
                x,
                sigmas,
                disable=False,
                callback=wrapped_callback,
                extra_args=extra_args,
            )
        else:
            raise NotImplementedError
