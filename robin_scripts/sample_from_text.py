import argparse
import os
import sys

import setGPU
import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
import clip

sys.path.append("..")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.mlp import MLP_mixer

txt_mean = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_txt_mean.npy")
txt_std = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_txt_std.npy")
img_mean_mlp = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_img_mean.npy")
img_std_mlp = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_img_mean.npy")

img_mean_dif = np.load("../../../../mlodata1/roazbind/imagenet64/train_embedding_mean.npy")
img_std_dif = np.load("../../../../mlodata1/roazbind/imagenet64/train_embedding_std.npy")

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.out_path)

    logger.log("loading CLIP encoder...")
    device = "cuda" if th.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    logger.log("encoding caption...")
    caption = clip.tokenize(args.caption).to(device)
    txt_emb = model.encode_text(caption).cpu().detach().numpy().squeeze()
    txt_emb = (txt_emb - txt_mean)/txt_std
    txt_emb = th.from_numpy(txt_emb).float().to(device)

    logger.log("loading MLP model...")
    checkpoint = th.load(args.mlp_checkpoint)
    model = MLP_mixer(emb_dim=args.emb_dim, width=512, num_layers=8, dropout=0.3).to(device)
    model.load_state_dict(checkpoint)

    logger.log("text embedding to image embedding...")
    img_emb = model(txt_emb).cpu().detach().numpy()
    img_emb = img_emb * img_std_mlp + img_mean_mlp
    img_emb = (img_emb - img_mean_dif)/img_std_dif
    img_emb = th.from_numpy(img_emb).float().to(device)

    logger.log("loading diffusion model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cuda:0")
    )
    model.to(dist_util.dev())
    model.eval()

    if args.image_guidance_path is not None:
        image_guidance = Image.open(args.image_guidance_path)
        image_guidance = np.array(image_guidance).transpose((2, 0, 1))
        image_guidance = image_guidance / 127.5 - 1
        image_guidance = th.from_numpy(image_guidance).to("cuda:0").float()
    else:
        image_guidance = None

    def model_fn(x, t, img_emb):
        if args.guidance_scale is None:
            return model(x, t, img_emb=img_emb)
        else:
            # Classifier-free guidance
            cond_output = model(x, t, img_emb=img_emb)
            uncond_output = model(x, t, img_emb=img_emb * 0)
            cond_eps, cond_var = th.split(cond_output, cond_output.shape[1] // 2, dim=1)
            uncond_eps, _ = th.split(uncond_output, uncond_output.shape[1] // 2, dim=1)
            eps = uncond_eps + float(args.guidance_scale) * (cond_eps - uncond_eps)
        return th.cat([eps, cond_var], dim=1)

    logger.log("sampling...")
    all_images = []
    model_kwargs = {}
    model_kwargs["img_emb"] = img_emb
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
            image_guidance_scale=args.image_guidance_scale
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
        caption="A picture of a dog.",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        out_path="",
        guidance_scale=None,
        image_guidance_path=None,
        image_guidance_scale=0.01,
        image_guidance_decay="linear",
        emb_dim=512,
        mlp_checkpoint="../log/MLP/mixer_fights_overfit/model_ES.pt"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()