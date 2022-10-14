# Master Thesis on Text-to-Image Generative Models
## Implementing and Experimenting with Diffusion Models for Text-to-Image Generation

*By **Robin Zbinden** under the supervision of **Luis Barba** and **Martin Jaggi***.

In this project, we **implement a text-to-image generative model** based on DALL-E 2 and conduct some experiments to understand the possibilities of this type of model. We also propose a new guidance method for diffusion models called *image guidance*. All the model specifications and results can be found in the `master_thesis_report.pdf`.

### How to generate images from text?

1. Download the checkpoints of the image decoder, CLIP translator, and upsampler, as well as the means and standard deviations of the embeddings [here](https://drive.google.com/drive/folders/1NEYwdgRLBx-nRvw66Td8cxr4nS5yTkgq?usp=sharing). Then put all these files into the folder named *models*.

2. Write a textual description of the images you want to generate in `captions.txt`. One caption per line.

3. Run the shell script to generate the images, i.e., `sh sample_from_text.sh`. Feel free to modify the number of samples generated per caption.

4. (Optional) Increase the resolution with the upsampler using the shell script `sample_upsampler.sh`. You need to specify the name of the `npz` file containing the 64x64 images with the argument `base_samples` in the script.

### Code

The code is divided into three folders: *guided_diffusion*, *scripts*, and *evaluations*. The other folder named *figures* contains the figures created for the master thesis report. The same seed (42) is used in all the experiments.

The code is based on [openai/guided-difusion](https://github.com/openai/guided-diffusion).

#### guided_diffusion

This folder contains all the methods to build our model, as well as helper functions to handle the datasets and to train. It is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). In particular, it consists of the following files (sorted by relevance):

- `gaussian_diffusion.py`: all the methods used to create and run diffusion processes.
- `unet.py`: the architecture definition of the U-Net diffusion model.
- `train_util.py`: helper functions to train the different models.
- `script_util.py`: helper functions for the scripts.
- `mlp.py`: the architecture definition of the CLIP translator.
- `losses.py`: the definitions of the different losses used to train the diffusion model.
- `dataset_helpers.py`: helper functions to handle the datasets.
- `nn.py`: basic neural network functions.
- `logger.py`: functions to log the different steps of training and sampling.
- `dist_util.py`: functions to distribute the training.
- `fp16_util.py`: functions to train in a 16 float precision (not used by our model).
- `resamples.py`: functions to change the distribution over the timesteps during training (not used by our model).
- `respace.py`: functions to respace the timesteps (not used by our model).

#### scripts

This folder contains the different scripts to train and sample from our method. A shell file is associated with each python script which requires many arguments. In particular, it consists of the following files (sorted by relevance):

- `sample_from_text.py`: generate images from a set of textual captions.
- `sample_upsampler.py`: increase the resolution of the images from 64x64 to 256x256.
- `sample_from_image.py`: generate images from an image embedding.
- `train_decoder.py`: train the image decoder.
- `train_translator.py`: train the CLIP translator.
- `clip_embeddings.py`: create the CLIP embeddings for a dataset.
- `handling_images.py`: create a figure from a set of images.

#### evaluations

This folder contains the methods to evaluate our method. Another `README.md` explaining the procedure to replicate the evaluations is available in this folder. 
