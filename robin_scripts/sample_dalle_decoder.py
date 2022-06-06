"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import sys

import setGPU
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

sys.path.append("..")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

EMBEDDING_PATH = "../../../../mlodata1/roazbind/imagenet64/train_embedding.npy"
EMBEDDING_MEAN_PATH = "../../../../mlodata1/roazbind/imagenet64/train_embedding_mean.npy"
EMBEDDING_STD_PATH = "../../../../mlodata1/roazbind/imagenet64/train_embedding_std.npy"

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.out_path)
    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cuda:0")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.image_guidance_path is not None:
        image_guidance = Image.open(args.image_guidance_path)
        image_guidance = np.array(image_guidance).transpose((2, 0, 1))
        image_guidance = image_guidance / 127.5 - 1
        image_guidance = th.from_numpy(image_guidance).to("cuda:0").float()
    else:
        image_guidance = None

    # img_emb = np.load("../images/other/corgi_hat_embedding.npy")
    img_emb = np.load(EMBEDDING_PATH)[args.img_id]

    mean = np.load(EMBEDDING_MEAN_PATH)   
    std = np.load(EMBEDDING_STD_PATH)
    img_emb = (img_emb - mean)/std
    img_emb = th.from_numpy(img_emb).to("cuda:0").float()

    def model_fn(x, t, img_emb, diffusion):
        if args.guidance_scale is None:
            return model(x, t, img_emb=img_emb)
        else:
            # Classifier-free guidance + dynamic thresholding
            cond_output = model(x, t, img_emb=img_emb)
            uncond_output = model(x, t, img_emb=img_emb * 0)
            cond_eps, cond_var = th.split(cond_output, cond_output.shape[1] // 2, dim=1)
            uncond_eps, _ = th.split(uncond_output, uncond_output.shape[1] // 2, dim=1)
            eps = uncond_eps + float(args.guidance_scale) * (cond_eps - uncond_eps)
            if args.dynamic_thresholding == True:
                x_0 = diffusion._predict_xstart_from_eps(x, t, eps)
                s = th.quantile(th.abs(x_0).flatten(1), 0.995, dim=1, keepdim=False)
                s = th.maximum(s, th.ones(s.shape).to("cuda:0"))[:, None, None, None]
                x_0 = th.clamp(x_0, -s, s) / s
                eps = diffusion._predict_eps_from_xstart(x, t, x_0)

        return th.cat([eps, cond_var], dim=1)

    logger.log("sampling...")
    all_images = []
    model_kwargs = {}
    model_kwargs["img_emb"] = img_emb
    model_kwargs["diffusion"] = diffusion
    while len(all_images) * args.batch_size < args.num_samples:
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None,
            device=dist_util.dev(),
            image_guidance=image_guidance,
            image_guidance_scale=args.image_guidance_scale,
            latent_save_interval=args.latent_save_interval
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        np.set_printoptions(threshold=np.inf)

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        for i, a in enumerate(arr):
            out_path = os.path.join(logger.get_dir(), f"sample_{i}.png")
            Image.fromarray(a,"RGB").save(out_path)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        out_path="",
        img_id=0,
        guidance_scale=None,
        dynamic_thresholding=False,
        image_guidance_path=None,
        image_guidance_scale=0.01,
        image_guidance_decay="linear",
        latent_save_interval=None,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
