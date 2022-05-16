cd ..

python sample_from_text.py \
    --captions "An image of a dog lying on the beach with birds flying in the sky. Birds flying in the sky above a beach with a dog." \
               "Second caption" \
    --num_samples 10 \
    --batch_size 10 \
    --use_ddim False \
    --model_path ../log/diffusion/diffusion_dalle_decoder_final_2_lr/ema_0.9999_800000.pt \
    --out_path ../images/pipeline/dog_birds \
    --guidance_scale 4 \
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
    #--image_guidance_path ../images/imagenet64/groundtruth/image0.jpg \
    #--image_guidance_scale 0.003 \

cd shell_files