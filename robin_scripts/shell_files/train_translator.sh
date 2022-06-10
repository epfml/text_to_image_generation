cd ..

python train_prior.py \
    --model_name "final_7" \
    --log_interval 50 \
    --save_interval 200000 \
    --batch_size 256 \
    --weight_decay 0.0002 \
    --dropout 0.1 \
    --epochs 100 \
    --emb_dim 512 \
    --num_layers 30 \

cd shell_files