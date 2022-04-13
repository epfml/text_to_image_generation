import torch
import clip
from PIL import Image
import numpy as np

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img_size = 64
img_size2 = img_size ** 2
for i in range(1, 6):

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
            image = preprocess(Image.fromarray(data[j], mode="RGB")).unsqueeze(0).to(device)
            embeddings.append(model.encode_image(image).cpu().numpy().squeeze())

    embeddings = np.vstack(embeddings)
    np.save(f"../../../../mlodata1/roazbind/imagenet64/embedding_batch_{i}.npy", embeddings)
    print(f"Batch {i} saved.")
