import os
import sys
import argparse

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append("..")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

num_captions = 5

class CocoDataset(Dataset):

    def __init__(self, txt_embeddings, img_embeddings, txt_mean, txt_std, img_mean, img_std):
        self.txt_embeddings = txt_embeddings
        self.img_embeddings = img_embeddings
        self.txt_mean = txt_mean
        self.txt_std = txt_std
        self.img_mean = img_mean
        self.img_std = img_std

    def __len__(self):
        return self.img_embeddings.shape[0]
    
    def __getitem__(self, ind):
        # Get a random caption among the available ones
        txt_embedding = self.txt_embeddings[ind][np.random.randint(num_captions)]
        txt_embedding = (txt_embedding - self.txt_mean)/self.txt_std
        img_embedding = self.img_embeddings[ind]
        img_embedding = (img_embedding - self.img_mean)/self.img_std
        return th.from_numpy(txt_embedding).float(), th.from_numpy(img_embedding).float()


class MLP(nn.Module):

    def __init__(self, emb_dim, width, num_layers, dropout=0.0):
        super(MLP, self).__init__()

        layers = []

        layers.append(nn.Linear(emb_dim, width))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers):
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(p=dropout))
    
        layers.append(nn.Linear(width, emb_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


class MLP_mixer(nn.Module):

    def __init__(self, emb_dim, width, num_layers, dropout=0.1):
        super(MLP_mixer, self).__init__()

        class PreNormResidual(nn.Module):
            def __init__(self, dim, fn):
                super().__init__()
                self.fn = fn
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                return self.fn(self.norm(x)) + x

        def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
            inner_dim = int(dim * expansion_factor)
            return nn.Sequential(
                dense(dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                dense(inner_dim, dim),
                nn.Dropout(dropout)
            )

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, width),
            *[
                PreNormResidual(width, FeedForward(width, 2, dropout)) for _ in range(num_layers)],
            nn.LayerNorm(width),
            nn.Linear(width, emb_dim)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():

    args = create_argparser().parse_args()
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    # dist_util.setup_dist()
    dir = f"../log/MLP/{args.model_name}"
    os.makedirs(dir, exist_ok=True)
    logger.configure(dir=dir, format_strs=["stdout","log","csv","tensorboard"])

    logger.log("loading dataset...")
    train_txt_embeddings = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_txt.npy")
    train_img_embeddings = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_img.npy")
    val_txt_embeddings = np.load(f"../../../../mlodata1/roazbind/coco/val_embedding_txt.npy")
    val_img_embeddings = np.load(f"../../../../mlodata1/roazbind/coco/val_embedding_img.npy")

    txt_mean = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_txt_mean.npy")
    txt_std = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_txt_std.npy")
    img_mean = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_img_mean.npy")
    img_std = np.load(f"../../../../mlodata1/roazbind/coco/train_embedding_img_mean.npy")

    train_set = CocoDataset(train_txt_embeddings, train_img_embeddings, 
                            txt_mean, txt_std, img_mean, img_std)
    test_set = CocoDataset(val_txt_embeddings, val_img_embeddings, 
                           txt_mean, txt_std, img_mean, img_std)

    batch_size = args.batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set, batch_size=len(test_set))
    val_txt_emb, val_img_emb = next(iter(test_loader))

    logger.log("creating model...")
    model = MLP_mixer(emb_dim=args.emb_dim, width=512, num_layers=8).to(device)
    
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
                    if loss < best_val_loss:
                        th.save(model.state_dict(), f"{args.model_name}/model_ES{(iteration):08d}.pt")

                # Identity model loss
                loss = criterion(val_txt_emb.to(device), val_img_emb.to(device))
                logger.logkv_mean("identity model loss", loss.item())

                logger.dumpkvs()

            if iteration % args.save_interval == 0:
                th.save(model.state_dict(), f"{args.model_name}/model{(iteration):08d}.pt")
                th.save(optimizer.state_dict(), f"{args.model_name}/opt{(iteration):08d}.pt")

            iteration += 1


def create_argparser():
    defaults = dict(
        model_name="model",
        log_interval=1000,
        save_interval=100000,
        batch_size=64,
        weight_decay=0.01,
        dropout=0.1,
        epochs = 10000,
        emb_dim=512
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()