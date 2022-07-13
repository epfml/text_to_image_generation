import argparse
import os
import shutil
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
    RANDOM_SEED
)
from guided_diffusion.mlp import MLP_mixer
from guided_diffusion.dataset_helpers import (
    EMBEDDING_IMAGE_MEAN_PATH,
    EMBEDDING_IMAGE_STD_PATH,
    EMBEDDING_CAPTION_MEAN_PATH,
    EMBEDDING_CAPTION_STD_PATH,
)

np.random.seed(RANDOM_SEED)
th.manual_seed(RANDOM_SEED)

txt_mean = np.load(EMBEDDING_CAPTION_MEAN_PATH)
txt_std = np.load(EMBEDDING_CAPTION_STD_PATH)
img_mean = np.load(EMBEDDING_IMAGE_MEAN_PATH)
img_std = np.load(EMBEDDING_IMAGE_STD_PATH)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.out_path)

    logger.log("loading CLIP encoder...")
    device = "cuda" if th.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    logger.log("encoding captions:")

    captions = []
    with open(args.captions_file) as f:
        for line in f:
            captions.append(line.strip('\n'))
    num_captions = len(captions)
    logger.log(captions)

    captions = clip.tokenize(captions).to(device)
    txt_embs = model.encode_text(captions).cpu().detach().numpy().squeeze()
    txt_embs = (txt_embs - txt_mean)/txt_std
    txt_embs = th.from_numpy(txt_embs).float().to(device)

    logger.log("loading MLP model...")
    checkpoint = th.load(args.mlp_checkpoint)
    model = MLP_mixer(emb_dim=args.emb_dim, width=512, num_layers=30, dropout=0.1).to(device)
    model.load_state_dict(checkpoint)

    logger.log("text embeddings to image embeddings...")
    img_embs = model(txt_embs).cpu().detach().numpy()
    img_embs = th.from_numpy(img_embs).float().to(device)

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

    if len(img_embs.shape) == 1:
        img_embs = img_embs[None, :]
    img_embs = th.repeat_interleave(img_embs, args.samples_per_caption, dim=0)

    all_images = []
    model_kwargs = {}
    model_kwargs["diffusion"] = diffusion
    i = 0
    while len(all_images) * args.batch_size < num_captions * args.samples_per_caption:

        remaining_samples = num_captions * args.samples_per_caption - (len(all_images) + 1) * args.batch_size
        if remaining_samples < 0:
            # Total number of images is not a multiple of batch_size
            batch_size = args.batch_size + remaining_samples
        else:
            batch_size = args.batch_size

        model_kwargs["img_emb"] = img_embs[i:i+batch_size]
        i += batch_size

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (batch_size, 3, args.image_size, args.image_size),
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
        logger.log(f"created {len(all_images) * args.batch_size - (args.batch_size - batch_size)} samples")

    arr = np.concatenate(all_images, axis=0)
    if dist.get_rank() == 0:
        
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        shutil.copy(args.captions_file, logger.get_dir())
        
        for j, a in enumerate(arr):
            out_path = os.path.join(logger.get_dir(), f"sample_{j}.png")
            Image.fromarray(a,"RGB").save(out_path)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        captions_file="captions.txt",
        clip_denoised=True,
        samples_per_caption=10,
        batch_size=16,
        use_ddim=False,
        model_path="",
        out_path="",
        guidance_scale=None,
        dynamic_thresholding=False,
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