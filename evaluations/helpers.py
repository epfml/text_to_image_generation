import math

import numpy as np
import torchvision.datasets as dset
from PIL import ImageOps


def process_coco(num_samples=50, new_size=256, random_seed=42, border=True):
    """Get num_samples images and captions from the COCO validation set. The images
       are of size new_size x new_size."""

    np.random.seed(random_seed)

    data = dset.CocoCaptions(root ="../../../../mlodata1/roazbind/coco/val2014", 
        annFile = '../../../../mlodata1/roazbind/coco/annotations/captions_val2014.json')

    samples_list = np.random.choice(len(data), num_samples)
    captions_index_list = np.random.randint(5, size=num_samples)
    captions_list = []
    images_list = []
    for i, j in enumerate(samples_list):
            image, captions = data[j]
            image = resize_image(image, new_size, border=border)
            image.save(f"samples/images{new_size}/image_{i}.png")
            images_list.append(np.array(image))
            captions_list.append(captions[captions_index_list[i]])

    np.savez(f"samples/val_images_{new_size}.npz", np.array(images_list))
    with open("samples/crop/val_captions.txt", 'w') as f:
        for caption in captions_list:
            f.write("%s\n" % caption)


def resize_image(image, new_size, border=True):
    """Resize image to new size. If border is True, add white borders to the image, otherwise center and crop the image."""

    width, height = image.size

    if border:
        max_length = max(width, height)

        # Add borders
        border = (math.ceil((max_length - width)/2), math.ceil((max_length - height)/2), 
                    math.floor((max_length - width)/2), math.floor((max_length - height)/2))
        image = ImageOps.expand(image, border=border, fill="white")

    else:
        min_length = min(width, height)

        # Center crop
        left = (width - min_length)/2
        top = (height - min_length)/2
        right = (width + min_length)/2
        bottom = (height + min_length)/2
        image = image.crop((left, top, right, bottom))

    # Resize
    image = image.resize((new_size, new_size))

    return image


if __name__ == "__main__":
    process_coco(200)