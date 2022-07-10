# Evaluating the model

We evaluate our model using the FID, Precision, Recall, and Inception Score. To compute them, two files are required: the one containing the generated images, e.g. `generated.npz`, 
as well as the one containing real images for the same captions e.g. `real.npz`. Then we simply run the `script.sh` file to compute the different metrics.

`helpers.py` contains the function `process_coco` which enables us to obtain random images and captions from the COCO validation set.
