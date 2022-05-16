python ../super_res_sample_test.py \
    --clip_denoised True \
    --num_samples 1 \
    --batch_size 1 \
    --use_ddim False \
    --base_samples ../images/imagenet64/image3_not_rotated_200000/samples_10x64x64x3.npz \
    --model_path ../openai_models/64_256_upsampler.pt \
    --out_path ../images/imagenet256/image3_final \
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
    --label 288 \
#    --use_checkpoint USE_CHECKPOINT \
#    --timestep_respacing TIMESTEP_RESPACING \

