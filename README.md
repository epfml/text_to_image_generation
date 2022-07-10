# Master Thesis on Text-to-Image Generative models
## Implementing and Experimenting with Diffusion Models for Text-to-Image Generation

*By **Robin Zbinden** under the supervision of **Luis Barba** and **Martin Jaggi***.

In this project, we **implement a text-to-image generative model** based on DALL-E 2 and conduct some experiments to understand the possibilities of this type of models. We also propose a new guidance method for diffusion models called *image guidance*. All the models specifications and results can be found in the `master_thesis_report.pdf`.

The code is divided into three folders: *guided_diffusion*, *scripts*, and *evaluations*. The other folder named *figures* contains the figures created for the master thesis report. The same seed (42) is used in all the experiments.


### guided_diffusion:

This folder contains all the methods to build our model, as well as helper functions to handle the datasets and to train. It is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). In particular, it consists of the following files (sorted by relevance):

- `gaussian_diffusion.py` contains all the methods used to create and run diffusion processes.
- `unet.py` contains the architecture definition of the Unet diffusion model.
- `train_util.py` contains helpers function to train the different models.
- `script_util.py` contains helpers function for the scripts.
- `mlp.py` contains the architecture definition of the CLIP translator.
- `losses.py` contains the definitions of the different losses used to train the diffusion model.
- `dataset_helpers.py` contains helpers function to handle the datasets.
- `nn.py` contains basic neural network functions.
- `logger.py` contains functions to log the different steps of training and sampling.
- `dist_util.py` contains functions to distribute the training.
- `fp16_util.py` contains functions to train in a 16 float precision (not used by our model).
- `resamples.py` contains functions to change the distribution over the timesteps during training (not used by our model).
- `respace.py` contains functions to respace the timesteps (not used by our model).

### scripts

This folder contains the different scripts to train and sample from our method. A shell file is associated to each python script which requires many arguments. In particular, it consists of the following files (sorted by relevance):

- `sample_from_text.py` generates images from a set of textual captions.
- `sample_upsampler.py` increases the resolution of the images from 64x64 to 256x256.
- `sample_from_image.py` generates images from an image embedding.
- `train_decoder.py` trains the image decoder.
- `train_translator.py` trains the CLIP translator.
- `clip_embeddings.py` creates the CLIP embeddings for a dataset.
- `handling_images.py` creates a figure from a set of images.

### evaluations
