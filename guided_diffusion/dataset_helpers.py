from bisect import bisect

import setGPU
import numpy as np
import webdataset as wds
import torch as th

from torch.utils.data import Dataset

EMBEDDING_IMAGE_MEAN_PATH = "../../../../mlodata1/roazbind/embeddings_stats/image_embedding_mean.npy"
EMBEDDING_IMAGE_STD_PATH = "../../../../mlodata1/roazbind/embeddings_stats/image_embedding_std.npy"


class ImagenetDataset(Dataset):

    def __init__(self, transform=None, normalize_emb=True, drop_emb_proba=0.2):

        if normalize_emb:
            self.mean = np.load(EMBEDDING_IMAGE_MEAN_PATH)   
            self.std = np.load(EMBEDDING_IMAGE_STD_PATH)
        self.normalize_emb = normalize_emb
        self.transform = transform
        self.drop_emb_proba = drop_emb_proba

        img_paths = [f"../../../../mlodata1/roazbind/imagenet64/train_data_batch_{i}.npy" for i in range(1, 11)]
        emb_paths = [f"../../../../mlodata1/roazbind/imagenet64/embedding_batch_{i}.npy" for i in range(1, 11)]

        self.img_memmaps = [np.load(path, mmap_mode='r') for path in img_paths]
        self.emb_memmaps = [np.load(path, mmap_mode='r') for path in emb_paths]

        self.start_indices = [0] * len(img_paths)
        self.img_count = 0
        for index, memmap in enumerate(self.img_memmaps):
            self.start_indices[index] = self.img_count
            self.img_count += memmap.shape[0]

    def __len__(self):
        return self.img_count

    def __getitem__(self, index):

        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        img = self.img_memmaps[memmap_index][index_in_memmap]
        emb = self.emb_memmaps[memmap_index][index_in_memmap]
        
        img = th.from_numpy(img)
        if self.transform is not None:
            img = self.transform(img)
        img = img / 127.5 - 1

        emb = th.from_numpy(emb).float()
        if self.normalize_emb:
            emb = (emb - self.mean)/self.std  
        if np.random.binomial(n=1, p=self.drop_emb_proba) == 1:
            emb *= 0

        return img, emb


class CCDataset(Dataset):

    def __init__(self, num_shard, image_folder_path, embeddings_folder_path, transform=None, normalize_emb=True, drop_emb_proba=0.2):

        if normalize_emb:
            self.mean = np.load(EMBEDDING_IMAGE_MEAN_PATH)   
            self.std = np.load(EMBEDDING_IMAGE_STD_PATH)
        self.normalize_emb = normalize_emb
        self.transform = transform
        self.drop_emb_proba = drop_emb_proba

        img_paths = [f"{image_folder_path}/0{str(i).zfill(4)}.npy" for i in range(1, num_shard + 1)]
        emb_paths = [f"{embeddings_folder_path}/0{str(i).zfill(4)}.npy" for i in range(1, num_shard + 1)]

        self.img_memmaps = [np.load(path, mmap_mode='r') for path in img_paths]
        self.emb_memmaps = [np.load(path, mmap_mode='r') for path in emb_paths]

        self.start_indices = [0] * len(img_paths)
        self.img_count = 0
        for index, memmap in enumerate(self.img_memmaps):
            self.start_indices[index] = self.img_count
            self.img_count += memmap.shape[0]

    def __len__(self):
        return self.img_count

    def __getitem__(self, index):

        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        img = self.img_memmaps[memmap_index][index_in_memmap]
        emb = self.emb_memmaps[memmap_index][index_in_memmap]
        
        img = th.from_numpy(img.transpose(2, 0, 1))
        if self.transform is not None:
            img = self.transform(img)
        img = img / 127.5 - 1

        emb = th.from_numpy(emb).float()
        if self.normalize_emb:
            emb = (emb - self.mean)/self.std  
        if np.random.binomial(n=1, p=self.drop_emb_proba) == 1:
            emb *= 0

        return img, emb


def get_iterator(dataloader):
        while True:
            yield from dataloader


def resize_images(dataset, num_shard, batch_size=512):
    """
    Resize images of dataset from size 256x256 to 64x64.
    """

    if dataset not in ["cc3m", "cc12m"]:
        raise Exception(f"Dataset {dataset} not supported.")
    
    with th.no_grad():
        for i in range(1, num_shard + 1):
            print(f"shard {i}")
            url = f"../../../../mlodata1/roazbind/{dataset}/images_256/0{str(i).zfill(4)}.tar"
            
            dataset = wds.WebDataset(url).decode("pil").to_tuple("jpg;png")

            dataloader = th.utils.data.DataLoader(dataset.batched(batch_size), num_workers=4, batch_size=None)
            
            resized = []
            for images in dataloader:
                resized.append([np.array(image.resize((64, 64))) for image in np.squeeze(images)])

            resized = np.concatenate(resized, axis=0)
            np.save(f"../../../../mlodata1/roazbind/{dataset}/images_64/0{str(i).zfill(4)}.npy", resized)