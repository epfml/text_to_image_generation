cd ..

python sample_dalle_decoder.py \
    --clip_denoised True \
    --num_samples 20 \
    --batch_size 20 \
    --use_ddim False \
    --model_path ../log/diffusion/final_3_1/ema_0.9999_500000.pt \
    --out_path .. \
    --img_id 3 \
    --guidance_scale 6 \
    --dynamic_thresholding True \
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
    --use_fp16 False \
    --use_new_attention_order False \
    --img_emb_dim 512 \
    --learn_sigma True \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --use_kl False \
    --predict_xstart False \
    --rescale_timesteps False \
    --rescale_learned_sigmas False \
    #--image_guidance_path ../images/other/corgi.png \
    #--image_guidance_scale 0.005 \
    #--image_guidance_decay linear \

cd shell_files
