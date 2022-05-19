import sys
import argparse

import setGPU
import torch
import clip
from PIL import Image
import numpy as np
import torchvision.datasets as dset
import webdataset as wds

sys.path.append("..")
from guided_diffusion.script_util import add_dict_to_argparser


# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def main():
    args = create_argparser().parse_args()

    datasets = {"imagenet": imagenet_emb,
                 "coco": coco_emb,
                 "cc3m": cc3m_emb,
                 "cc12m": cc12m_emb}

    if args.dataset in datasets.keys():
        datasets[args.dataset]()
    else:
        raise Exception(f"Dataset {args.dataset} is not available, take one in {datasets.keys()} instead.")


def imagenet_emb():

    img_size = 64
    img_size2 = img_size ** 2
    for i in range(0, 11):

        print(f"\nProcess batch {i}")
        batch = np.load(f"../../../../mlodata1/roazbind/imagenet64/train_data_batch_{i}.npz")

        # Reshape data to NHWC
        data = batch["data"]
        data = np.dstack((data[:, :img_size2], data[:, img_size2:2*img_size2], data[:, 2*img_size2:]))
        data = data.reshape((data.shape[0], img_size, img_size, 3))
        print(f"Data with shape: {data.shape}")

        embeddings = []
        with torch.no_grad():
            for j in range(len(data)):
                if j % 25000 == 0:
                    print(f"Process image {j + 1}")
                image = Image.fromarray(data[j], mode="RGB")
                embeddings.append(get_image_embedding(image))

        embeddings = np.vstack(embeddings)
        np.save(f"../../../../mlodata1/roazbind/imagenet64/embedding_batch_{i}.npy", embeddings)
        print(f"Batch {i} saved.")


def coco_emb():

    print("Processing train set")
    data = dset.CocoCaptions(root ="../../../../mlodata1/roazbind/coco/train2014", 
                            annFile = '../../../../mlodata1/roazbind/coco/annotations/captions_train2014.json')

    img_embeddings = []
    txt_embeddings = []
    with torch.no_grad():
        for j in range(len(data)):

            if j % 10000 == 0:
                print(f"Process image {j + 1}")
            image, captions = data[j]
            img_embeddings.append(get_image_embedding(image))
            txt_embeddings.append(get_text_embedding(captions[:5]))

    img_embeddings = np.array(img_embeddings)
    txt_embeddings = np.array(txt_embeddings)

    np.save(f"../../../../mlodata1/roazbind/coco/train_embedding_img.npy", img_embeddings)
    np.save(f"../../../../mlodata1/roazbind/coco/train_embedding_txt.npy", txt_embeddings)

    np.save(f"../../../../mlodata1/roazbind/coco/train_embedding_img_mean.npy", np.mean(img_embeddings, 0))
    np.save(f"../../../../mlodata1/roazbind/coco/train_embedding_img_std.npy", np.std(img_embeddings, 0))
    np.save(f"../../../../mlodata1/roazbind/coco/train_embedding_txt_mean.npy", np.mean(txt_embeddings, (0,1)))
    np.save(f"../../../../mlodata1/roazbind/coco/train_embedding_txt_std.npy", np.std(txt_embeddings, (0,1)))

    print("Processing validation set")
    data = dset.CocoCaptions(root ="../../../../mlodata1/roazbind/coco/val2014", 
                            annFile = '../../../../mlodata1/roazbind/coco/annotations/captions_val2014.json')

    img_embeddings = []
    txt_embeddings = []
    with torch.no_grad():
        for j in range(len(data)):
            if j % 5000 == 0:
                print(f"Process image {j + 1}")
            image, captions = data[j]
            img_embeddings.append(get_image_embedding(image))
            txt_embeddings.append(get_text_embedding(captions[:5]))

    img_embeddings = np.array(img_embeddings)
    txt_embeddings = np.array(txt_embeddings)

    np.save(f"../../../../mlodata1/roazbind/coco/val_embedding_img.npy", img_embeddings)
    np.save(f"../../../../mlodata1/roazbind/coco/val_embedding_txt.npy", txt_embeddings)


def cc3m_emb():

    num_shard = 331
    num_samples = 0
    with torch.no_grad():
        for i in range(1, num_shard + 1):
            print(f"shard {i}")
            url = f"../../../../mlodata1/roazbind/cc3m/images_256/00{str(i).zfill(3)}.tar"
            
            img_embeddings, txt_embeddings = ccm_helper(url)
      
            num_samples += len(img_embeddings)
            print(len(img_embeddings))

            np.save(f"../../../../mlodata1/roazbind/cc3m/embeddings/captions/00{str(i).zfill(3)}.npy", txt_embeddings)
            np.save(f"../../../../mlodata1/roazbind/cc3m/embeddings/images/00{str(i).zfill(3)}.npy", img_embeddings)
    print(num_samples)


def cc12m_emb():

    num_shard = 1242
    num_samples = 0
    with torch.no_grad():
        for i in range(1, num_shard + 1):
            print(f"shard {i}")
            url = f"../../../../mlodata1/roazbind/cc12m/images_256/00{str(i).zfill(3)}.tar"
            
            img_embeddings, txt_embeddings = ccm_helper(url)
      
            num_samples += len(img_embeddings)
            print(len(img_embeddings))

            np.save(f"../../../../mlodata1/roazbind/cc12m/embeddings/captions/00{str(i).zfill(3)}.npy", txt_embeddings)
            np.save(f"../../../../mlodata1/roazbind/cc12m/embeddings/images/00{str(i).zfill(3)}.npy", img_embeddings)
    print(num_samples)


def ccm_helper(url):

    dataset = wds.WebDataset(url).decode("pil").to_tuple("jpg;png", "json", "txt").map_tuple(preprocess, identity, identity)
            
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size), num_workers=4, batch_size=None)

    img_embeddings = []
    txt_embeddings = []
    for images, _, captions in dataloader:
        img_embeddings.append(model.encode_image(images.to(device)).cpu().numpy().squeeze())
        txt_embeddings.append(get_text_embedding(captions))

    img_embeddings = np.concatenate(img_embeddings, axis=0)
    txt_embeddings = np.concatenate(txt_embeddings, axis=0)

    return img_embeddings, txt_embeddings


def identity(x):
        return x


def get_image_embedding(image):
    image = preprocess(image).unsqueeze(0).to(device)
    return model.encode_image(image).cpu().numpy().squeeze()


def get_text_embedding(captions):
    captions = clip.tokenize(captions, truncate=True).to(device)
    return model.encode_text(captions).cpu().numpy().squeeze()


def create_argparser():
        defaults = dict(
            dataset="imagenet"
        )
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser


if __name__ == "__main__":
    main()
    
