import setGPU
import numpy as np
import webdataset as wds
import torch


def resize_images(dataset, num_shard, batch_size=512):
    """
    Resize images of dataset from size 256x256 to 64x64.
    """

    if dataset not in ["cc3m", "cc12m"]:
        raise Exception(f"Dataset {dataset} not supported.")
    
    with torch.no_grad():
        for i in range(1, num_shard + 1):
            print(f"shard {i}")
            url = f"../../../../mlodata1/roazbind/{dataset}/images_256/0{str(i).zfill(4)}.tar"
            
            dataset = wds.WebDataset(url).decode("pil").to_tuple("jpg;png")

            dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size), num_workers=4, batch_size=None)
            
            resized = []
            for images in dataloader:
                resized.append([np.array(image.resize((64, 64))) for image in np.squeeze(images)])

            resized = np.concatenate(resized, axis=0)
            np.save(f"../../../../mlodata1/roazbind/{dataset}/images_64/0{str(i).zfill(4)}.npy", resized)
