import os
import sys
import argparse

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append("..")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser
)
from guided_diffusion.mlp import MLP_mixer
from guided_diffusion.dataset_helpers import CCCaptionsDataset, CocoDataset

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

num_captions = 5


def main():

    args = create_argparser().parse_args()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    # dist_util.setup_dist()
    dir = f"../log/MLP/{args.model_name}"
    os.makedirs(dir, exist_ok=True)
    logger.configure(dir=dir, format_strs=["stdout","log","csv","tensorboard"])

    logger.log("loading datasets...")

    logger.log("loading cc3m dataset...")
    num_shard = 331
    img_emb_folder_path = "../../../../mlodata1/roazbind/cc3m/embeddings/images"
    txt_emb_folder_path = "../../../../mlodata1/roazbind/cc3m/embeddings/captions"
    cc3m_dataset = CCCaptionsDataset(num_shard, img_emb_folder_path, txt_emb_folder_path)

    logger.log("loading cc12m dataset...")
    num_shard = 125 # less shards, before 1242
    img_emb_folder_path = "../../../../mlodata1/roazbind/cc12m/embeddings/images_less_shards"
    txt_emb_folder_path = "../../../../mlodata1/roazbind/cc12m/embeddings/captions_less_shards"
    cc12m_dataset = CCCaptionsDataset(num_shard, img_emb_folder_path, txt_emb_folder_path)

    logger.log("loading COCO dataset...")
    train_txt_embeddings = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_txt.npy")
    train_img_embeddings = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_img.npy")
    val_txt_embeddings = np.load(f"../../../../mlodata1/roazbind/coco/val_embedding_txt.npy")
    val_img_embeddings = np.load(f"../../../../mlodata1/roazbind/coco/val_embedding_img.npy")
    coco_train_set = CocoDataset(train_txt_embeddings, train_img_embeddings)
    test_set = CocoDataset(val_txt_embeddings, val_img_embeddings)

    train_set = ConcatDataset((cc3m_dataset, cc12m_dataset, coco_train_set))

    batch_size = args.batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=len(test_set))
    val_txt_emb, val_img_emb = next(iter(test_loader))

    logger.log("creating model...")
    model = MLP_mixer(emb_dim=args.emb_dim, width=512, num_layers=args.num_layers).to(device)
    
    optimizer = th.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    logger.log("training...")
    model.train()
    iteration = 0
    for epoch in range(args.epochs):
        for data in train_loader:
            optimizer.zero_grad()
            txt_emb, img_emb = data
            output = model(txt_emb.to(device))
            loss = criterion(output, img_emb.to(device))
            loss.backward()
            optimizer.step()

            logger.logkv_mean("train loss", loss.item())
            logger.logkv_mean("iteration", iteration)
            logger.logkv_mean("samples", iteration * batch_size)
            logger.logkv_mean("epoch", epoch)

            if iteration % args.log_interval == 0:

                # Compute validation loss
                with th.no_grad():
                    output = model(val_txt_emb.to(device))
                    loss = criterion(output, val_img_emb.to(device))
                    logger.logkv_mean("validation loss", loss.item())
                    if loss.item() < best_val_loss:
                        logger.log(f"Save model at epoch {epoch} and iteration {iteration} with val loss {loss.item()}")
                        th.save(model.state_dict(), f"{dir}/model_ES.pt")
                        best_val_loss = loss.item()

                # Identity model loss
                loss = criterion(val_txt_emb.to(device), val_img_emb.to(device))
                logger.logkv_mean("identity model loss", loss.item())

                logger.dumpkvs()

            if iteration % args.save_interval == 0 and iteration != 0:
                th.save(model.state_dict(), f"{dir}/model{(iteration):08d}.pt")
                th.save(optimizer.state_dict(), f"{dir}/opt{(iteration):08d}.pt")

            iteration += 1


def create_argparser():
    defaults = dict(
        model_name="model",
        log_interval=1000,
        save_interval=10000,
        batch_size=256,
        weight_decay=0.04,
        dropout=0.3,
        epochs=1000,
        num_layers=8,
        emb_dim=512
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()