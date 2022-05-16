cd ..

python train_prior.py \
    --model_name "mixer_fights_overfit" \
    --log_interval 1000 \
    --save_interval 10000 \
    --batch_size 256 \
    --weight_decay 0.04 \
    --dropout 0.3 \
    --epochs 1000 \
    --emb_dim 512 \

cd shell_files