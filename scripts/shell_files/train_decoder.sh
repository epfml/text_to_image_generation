cd ..

python train_decoder.py \
    --schedule_sampler uniform \
    --lr 3e-4 \
    --weight_decay 0 \
    --lr_anneal_steps 800000 \
    --batch_size 16 \
    --microbatch -1 \
    --ema_rate 0.9999 \
    --log_interval 10 \
    --save_interval 100000 \
    --use_fp16 False \
    --fp16_scale_growth 1e-3 \
    --gradient_clipping 0.008 \
    --model_name final_3_1 \
    --image_size 64 \
    --num_channels 256 \
    --num_res_blocks 3 \
    --num_heads -1 \
    --num_heads_upsample -1 \
    --num_head_channels 64 \
    --attention_resolutions 32,16,8 \
    --channel_mult 1,2,3,4 \
    --dropout 0.1 \
    --class_cond False \
    --emb_cond True \
    --use_checkpoint False \
    --use_scale_shift_norm True \
    --resblock_updown True \
    --use_new_attention_order False \
    --img_emb_dim 512 \
    --learn_sigma True \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --use_kl False \
    --predict_xstart False \
    --rescale_timesteps False \
    --rescale_learned_sigmas False \
    #--resume_checkpoint ../log/diffusion/final_3/model100000.pt

cd shell_files
