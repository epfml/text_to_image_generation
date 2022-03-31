# Script to train a model on cifar100

source activate robin_env

: '
python train_cifar100.py \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --lr 1e-4 \
    --batch_size 64 \
    --class_cond True \
    --learn_sigma True \
    --lr_anneal_steps 5000 \
'

python train_dif_cifar.py \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --lr 3e-4 \
    --batch_size 64 \
    --class_cond True \
    --learn_sigma True \
    --lr_anneal_steps 5000 \
    --weight_decay 0.05 \
    --use_fp16 True \
    --model_name diffusion_cifar100_test_tb \