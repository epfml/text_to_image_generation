"""
Train a diffusion model on images from cifar100.
"""

import os
import sys
import argparse

import setGPU

import torch as th
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append("..")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.dataset_helpers import ImagenetDataset, CCDataset, get_iterator

th.manual_seed(42)


def main():

    args = create_argparser().parse_args()
    dist_util.setup_dist()
    dir = f"../log/diffusion/{args.model_name}"
    os.makedirs(dir, exist_ok=True)
    logger.configure(dir=dir, format_strs=["stdout","log","csv","tensorboard"])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("loading dataset...")

    transform = [transforms.RandomHorizontalFlip(p=0.5)]#, transforms.RandomCrop(64, padding=4)]
    transform = transforms.Compose(transform)

    logger.log("loading cc3m dataset...")
    num_shard = 331
    image_folder_path = "../../../../mlodata1/roazbind/cc3m/images_64"
    embeddings_folder_path = "../../../../mlodata1/roazbind/cc3m/embeddings/images"
    cc3m_dataset = CCDataset(num_shard, image_folder_path, embeddings_folder_path, transform=transform)

    logger.log("loading cc12m dataset...")
    num_shard = 1242
    image_folder_path = "../../../../mlodata1/roazbind/cc12m/images_64"
    embeddings_folder_path = "../../../../mlodata1/roazbind/cc12m/embeddings/images"
    cc12m_dataset = CCDataset(num_shard, image_folder_path, embeddings_folder_path, transform=transform)

    logger.log("loading ImageNet dataset...")
    imagenet = ImagenetDataset(transform=transform)
    
    dataset = ConcatDataset((cc3m_dataset, cc12m_dataset, imagenet))
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
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
