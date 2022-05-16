# Script to train and sample from a model on cifar100

# Train diffusion model
: '
python train_dif_cifar.py \
    --schedule_sampler uniform \
    --lr 3e-4 \
    --weight_decay 0 \
    --lr_anneal_steps 15000 \
    --batch_size 128 \
    --microbatch -1 \
    --ema_rate 0.9999 \
    --log_interval 10 \
    --save_interval 7500 \
    --use_fp16 True \
    --fp16_scale_growth 1e-3 \
    --model_name diffusion_cifar100_linear_2 \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --num_heads -1 \
    --num_heads_upsample -1 \
    --num_head_channels 64 \
    --attention_resolutions 16,8,4 \
    --channel_mult 1,2,3,4 \
    --dropout 0 \
    --class_cond True \
    --use_checkpoint False \
    --use_scale_shift_norm True \
    --resblock_updown True \
    --use_new_attention_order False \
    --num_classes 100 \
    --learn_sigma True \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --use_kl False \
    --predict_xstart False \
    --rescale_timesteps False \
    --rescale_learned_sigmas False \
'
#    --data_dir \
#    --resume_checkpoint \
#    --timestep_respacing  \


# Train classifier model
: '
python train_cla_cifar.py \
    --noised True \
    --iterations 50000 \
    --lr 6e-4 \
    --weight_decay 0.2 \
    --anneal_lr False \
    --batch_size 128 \
    --microbatch -1 \
    --schedule_sampler uniform \
    --log_interval 10 \
    --eval_interval 5 \
    --save_interval 25000 \
    --model_name classifier_cifar100_linear_fast \
    --image_size 32 \
    --classifier_use_fp16 False \
    --classifier_width 128 \
    --classifier_depth 4 \
    --classifier_attention_resolutions 16,8,4 \
    --classifier_use_scale_shift_norm True \
    --classifier_resblock_updown True \
    --classifier_pool attention \
    --num_classes 100 \
    --learn_sigma True \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --use_kl False \
    --predict_xstart False \
    --rescale_timesteps False \
    --rescale_learned_sigmas False \
#    --data_dir 
#    --val_data_dir
#    --resume_checkpoint
#    --timestep_respacing
'

# Sample from the model

python sample_cifar.py \
    --clip_denoised True \
    --num_samples 10 \
    --batch_size 10 \
    --use_ddim False \
    --model_path ../log/diffusion_cifar100/model015000.pt \
    --classifier_path ../log/classifier_cifar100/model025000.pt \
    --classifier_scale 1 \
    --out_path ../images/cifar100/class_2 \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --num_heads -1 \
    --num_heads_upsample -1 \
    --num_head_channels 64 \
    --attention_resolutions 16,8,4 \
    --channel_mult 1,2,3,4 \
    --dropout 0 \
    --class_cond True \
    --use_checkpoint False \
    --use_scale_shift_norm True \
    --resblock_updown True \
    --use_fp16 True \
    --use_new_attention_order False \
    --num_classes 100 \
    --learn_sigma True \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --use_kl False \
    --predict_xstart False \
    --rescale_timesteps False \
    --rescale_learned_sigmas False \
    --classifier_use_fp16 False \
    --classifier_width 128 \
    --classifier_depth 4 \
    --classifier_attention_resolutions 16,8,4 \
    --classifier_use_scale_shift_norm True \
    --classifier_resblock_updown True \
    --classifier_pool attention \
#    --timestep_respacing
