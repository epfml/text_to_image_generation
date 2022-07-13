cd ..

python train_translator.py \
    --model_name "final" \
    --log_interval 500 \
    --save_interval 200000 \
    --epochs 6 \
    --batch_size 256 \
    --weight_decay 0.0001 \
    --dropout 0.1 \
    --emb_dim 512 \
    --num_layers 30 \
    --use_coco False \

cd shell_files