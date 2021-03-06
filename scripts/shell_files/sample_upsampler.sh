cd ..

python sample_upsampler.py \
    --base_samples ../samples_10x64x64x3.npz \
    --num_samples 10 \
    --batch_size 10 \
    --use_ddim False \
    --model_path ../models/upsampler.pt \
    --out_path .. \
    --num_channels 192 \
    --num_res_blocks 2 \
    --num_heads 4 \
    --num_heads_upsample -1 \
    --num_head_channels -1 \
    --attention_resolutions 32,16,8 \
    --dropout 0.0 \
    --class_cond True \
    --use_scale_shift_norm True \
    --resblock_updown True \
    --use_fp16 False \
    --num_classes 1000 \
    --learn_sigma True \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --use_kl False \
    --predict_xstart False \
    --rescale_timesteps False \
    --rescale_learned_sigmas False \
    --large_size 256 \
    --small_size 64 \
    --label 436 \
    --clip_denoised True \
#    --use_checkpoint USE_CHECKPOINT \
#    --timestep_respacing TIMESTEP_RESPACING \

cd shell_files
