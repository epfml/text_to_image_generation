from bisect import bisect

import setGPU
import numpy as np
import webdataset as wds
import torch as th

from torch.utils.data import Dataset


# Embeddings data paths
EMBEDDING_IMAGE_MEAN_PATH = "../models/image_embedding_mean.npy"
EMBEDDING_IMAGE_STD_PATH = "../models/image_embedding_std.npy"
EMBEDDING_CAPTION_MEAN_PATH = "../models/caption_embedding_mean.npy"
EMBEDDING_CAPTION_STD_PATH = "../models/caption_embedding_std.npy"

EMBEDDING_IMAGE_COCOVAL_PATH = "../../../../mlodata1/roazbind/coco/val_embedding_img.npy"
EMBEDDING_IMAGE_IMAGENET_BATCH_PATH = lambda i : \
                f"../../../../mlodata1/roazbind/imagenet64/embedding_batch_{i}.npy"

# Images data paths
IMAGES_IMAGENET_BATCH_PATH = lambda i : \
                f"../../../../mlodata1/roazbind/imagenet64/train_data_batch_{i}.npy"
IMAGES_CC_PATH = lambda dataset_name, image_size, i : \
                f"../../../../mlodata1/roazbind/{dataset_name}/images_{image_size}/0{str(i).zfill(4)}.tar"


# --------- Datasets of images ---------

class ImagenetDataset(Dataset):

    """
    Create the ImageNet dataset.

    :param transform: transformation applied to each image.
    :param normalize_emb: if True, the embeddings are normalized.
    :param drop_emb_proba: probability to drop the embedding, i.e., to
                            set the embedding to the null vector.
    :return: the ImageNet dataset.
    """
    def __init__(self, transform=None, normalize_emb=True, drop_emb_proba=0.2):

        if normalize_emb:
            self.mean = np.load(EMBEDDING_IMAGE_MEAN_PATH)   
            self.std = np.load(EMBEDDING_IMAGE_STD_PATH)
        self.normalize_emb = normalize_emb
        self.transform = transform
        self.drop_emb_proba = drop_emb_proba

        img_paths = [IMAGES_IMAGENET_BATCH_PATH(i) for i in range(1, 11)]
        emb_paths = [EMBEDDING_IMAGE_IMAGENET_BATCH_PATH(i) for i in range(1, 11)]

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
        """Get the image and its corresponding embedding of the specified index."""

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

    """
    Create a Conceptual Captions (CC) dataset by only considering the images
    and not the captions.

    :param num_shard: number of shards used to store the dataset.
    :param image_folder_path: path of the folder containing the images.
    :param embeddings_folder_path: path of the folder containing the image embeddings.
    :param transform: transformation applied to each image.
    :param normalize_emb: if True, the embeddings are normalized.
    :param drop_emb_proba: probability to drop the embedding, i.e., to
                            set the embedding to the null vector.
    :return: a Conceptual Captions (CC) dataset.
    """
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
        """Get the image and its corresponding embedding of the specified index."""

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


# --------- Datasets of embeddings pairs ---------

class CCCaptionsDataset(Dataset):

    """
    Create a Conceptual Captions (CC) dataset by considering the image
    and text embeddings only.

    :param num_shard: number of shards used to store the dataset.
    :param img_emb_folder_path: path of the folder containing the image embeddings.
    :param txt_emb_folder_path: path of the folder containing the text embeddings.
    :return: a Conceptual Captions (CC) dataset.
    """
    def __init__(self, num_shard, img_emb_folder_path, txt_emb_folder_path):

        
        self.img_emb_mean = np.load(EMBEDDING_IMAGE_MEAN_PATH)   
        self.img_emb_std = np.load(EMBEDDING_IMAGE_STD_PATH)
        self.txt_emb_mean = np.load(EMBEDDING_CAPTION_MEAN_PATH)   
        self.txt_emb_std = np.load(EMBEDDING_CAPTION_STD_PATH)

        img_emb_paths = [f"{img_emb_folder_path}/0{str(i).zfill(4)}.npy" for i in range(1, num_shard + 1)]
        txt_emb_paths = [f"{txt_emb_folder_path}/0{str(i).zfill(4)}.npy" for i in range(1, num_shard + 1)]

        self.img_emb_memmaps = [np.load(path, mmap_mode='r') for path in img_emb_paths]
        self.txt_emb_memmaps = [np.load(path, mmap_mode='r') for path in txt_emb_paths]

        self.start_indices = [0] * len(img_emb_paths)
        self.emb_count = 0
        for index, memmap in enumerate(self.img_emb_memmaps):
            self.start_indices[index] = self.emb_count
            self.emb_count += memmap.shape[0]

    def __len__(self):
        return self.emb_count

    def __getitem__(self, index):
        """Get the image and text embedding of the specified index."""

        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        img_emb = self.img_emb_memmaps[memmap_index][index_in_memmap]
        txt_emb = self.txt_emb_memmaps[memmap_index][index_in_memmap]
        
        img_emb = th.from_numpy(img_emb)
        img_emb = (img_emb - self.img_emb_mean)/self.img_emb_std  

        txt_emb = th.from_numpy(txt_emb).float()
        txt_emb = (txt_emb - self.txt_emb_mean)/self.txt_emb_std  

        return img_emb, txt_emb


class CocoDataset(Dataset):

    """
    Create the MS-COCO dataset by considering the image
    and text embeddings only.

    :param txt_embeddings: numpy array containing the text embeddings.
    :param img_embeddings: numpy array containing the image embeddings.
    :return: the MS-COCO dataset
    """
    def __init__(self, txt_embeddings, img_embeddings):
        self.num_captions = 5
        self.txt_embeddings = txt_embeddings
        self.img_embeddings = img_embeddings

        self.img_emb_mean = np.load(EMBEDDING_IMAGE_MEAN_PATH)   
        self.img_emb_std = np.load(EMBEDDING_IMAGE_STD_PATH)
        self.txt_emb_mean = np.load(EMBEDDING_CAPTION_MEAN_PATH)   
        self.txt_emb_std = np.load(EMBEDDING_CAPTION_STD_PATH)

    def __len__(self):
        return self.img_embeddings.shape[0]
    
    def __getitem__(self, ind):
        """Get the image and text embedding of the specified index."""

        # Get a random caption among the available ones
        txt_embedding = self.txt_embeddings[ind][np.random.randint(self.num_captions)]
        txt_embedding = (txt_embedding - self.txt_emb_mean)/self.txt_emb_std
        img_embedding = self.img_embeddings[ind]
        img_embedding = (img_embedding - self.img_emb_mean)/self.img_emb_std
        return th.from_numpy(txt_embedding).float(), th.from_numpy(img_embedding).float()


def get_iterator(dataloader):
    """Create an iterator from the dataloader."""
    while True:
        yield from dataloader


def resize_images(dataset, num_shard, batch_size=512):
    """
    Resize images of dataset from size 256x256 to 64x64 for the 
    Conceptual Captions (cc3m and cc12m) datasets.
    """

    if dataset not in ["cc3m", "cc12m"]:
        raise Exception(f"Dataset {dataset} not supported.")
    
    with th.no_grad():
        for i in range(1, num_shard + 1):
            print(f"shard {i}")
            url = IMAGES_CC_PATH(dataset_name=dataset, image_size=256, i=i)
            
            dataset = wds.WebDataset(url).decode("pil").to_tuple("jpg;png")

            dataloader = th.utils.data.DataLoader(dataset.batched(batch_size), num_workers=4, batch_size=None)
            
            resized = []
            for images in dataloader:
                resized.append([np.array(image.resize((64, 64))) for image in np.squeeze(images)])

            resized = np.concatenate(resized, axis=0)
            np.save(IMAGES_CC_PATH(dataset_name=dataset, image_size=64, i=i), resized)
