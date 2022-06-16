import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

indices = np.arange(0,12)

fig = plt.figure(figsize=(3, 4))
columns = 3
rows = 4
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    # img = mpimg.imread(f'../evaluations/samples/images256/image_{indices[i - 1]}.png')
    # img = mpimg.imread(f'../images/pipeline256/coco_50/sample_{indices[i - 1]}.png')
    img = mpimg.imread(f'../images/coco/img_emb/test_2_256/sample_{indices[i - 1]}.png')

    plt.axis('off')
    plt.margins(0)
    plt.imshow(img)

fig.tight_layout()

plt.subplots_adjust(bottom=0, left=0, right=1, top=1, hspace = 0, wspace = 0)
plt.margins(0)

plt.savefig("../figures/decoder.jpg", bbox_inches='tight', pad_inches=0)
