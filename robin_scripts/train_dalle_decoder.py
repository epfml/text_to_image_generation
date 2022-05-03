"""
Train a diffusion model on images from cifar100.
"""

import os
import sys
import argparse

import setGPU

import numpy as np
import torch as th
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

sys.path.append("..")
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

IMAGES_PATH = "../../../../mlodata1/roazbind/imagenet64/train_data.npz"
EMBEDDING_PATH = "../../../../mlodata1/roazbind/imagenet64/train_embedding.npy"
EMBEDDING_MEAN_PATH = "../../../../mlodata1/roazbind/imagenet64/train_embedding_mean.npy"
EMBEDDING_STD_PATH = "../../../../mlodata1/roazbind/imagenet64/train_embedding_std.npy"


class ImgEmbPair(Dataset):
    def __init__(self, img_path, emb_path, img_size=64, transform=None, normalize_emb=True, drop_emb_proba=0.2):

        data = np.load(img_path)["data"]

        # Reshape data to NCHW
        img_size2 = img_size**2
        data = np.dstack((data[:, :img_size2], data[:, img_size2:2*img_size2], data[:, 2*img_size2:]))
        self.imgs = data.reshape((data.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

        self.embs = np.load(emb_path)

        if normalize_emb:
            self.mean = np.load(EMBEDDING_MEAN_PATH)   
            self.std = np.load(EMBEDDING_STD_PATH)
        self.normalize_emb = normalize_emb
        self.transform = transform
        self.drop_emb_proba = drop_emb_proba

        assert len(self.imgs) == len(self.embs)

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):

        img = self.imgs[idx]
        img = th.from_numpy(img)
        if self.transform is not None:
            img = self.transform(img)
        img = img / 127.5 - 1

        emb = self.embs[idx]
        emb = th.from_numpy(emb).float()
        if self.normalize_emb:
            emb = (emb - self.mean)/self.std  
        if np.random.binomial(n=1, p=self.drop_emb_proba) == 1:
            emb *= 0

        return img, emb


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    dir = f"../log/{args.model_name}"
    os.makedirs(dir, exist_ok=True)
    logger.configure(dir=dir, format_strs=["stdout","log","csv","tensorboard"])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("loading dataset")

    transform = [transforms.RandomHorizontalFlip(p=0.5)]#, transforms.RandomCrop(64, padding=4)]
    transform = transforms.Compose(transform)

    dataset = ImgEmbPair(IMAGES_PATH, EMBEDDING_PATH, transform=transform)
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True)

    def get_iterator(dataloader):
        while True:
            yield from dataloader

    data = get_iterator(data)
    
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate, 
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_name=args.model_name,
        emb_cond=args.emb_cond,
        gradient_clipping=args.gradient_clipping
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_name="model",
        gradient_clipping=None
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
