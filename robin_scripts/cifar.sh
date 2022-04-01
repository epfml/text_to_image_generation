# Script to train and sample from a model on cifar100
: '
python train_dif_cifar.py \
    --data_dir
    --schedule_sampler
    --lr
    --weight_decay
    --lr_anneal_steps
    --batch_size
    --microbatch
    --ema_rate
    --log_interval
    --save_interval
    --resume_checkpoint
    --use_fp16
    --fp16_scale_growth
    --model_name
    --image_size
    --num_channels
    --num_res_blocks
    --num_heads
    --num_heads_upsample
    --num_head_channels
    --attention_resolutions
    --channel_mult
    --dropout
    --class_cond
    --use_checkpoint
    --use_scale_shift_norm
    --resblock_updown
    --use_new_attention_order
    --num_classes
    --learn_sigma
    --diffusion_steps
    --noise_schedule
    --timestep_respacing
    --use_kl
    --predict_xstart
    --rescale_timesteps
    --rescale_learned_sigmas
'
## OLD

# Train the diffusion model
: '
python train_dif_cifar.py \
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

: '
python train_dif_cifar.py \
    --image_size 32 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --diffusion_steps 1000 \
    --noise_schedule cosine \
    --lr 3e-4 \
    --batch_size 64 \
    --class_cond True \
    --num_classes 100 \
    --learn_sigma True \
    --lr_anneal_steps 5000 \
    --weight_decay 0.05 \
    --use_fp16 True \
    --model_name diffusion_cifar100_test_code \
'


# Train the classifier model
python train_cla_cifar.py \
    --image_size 32 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --lr 3e-4 \
    --batch_size 128 \
    --learn_sigma True \
    --iterations 5000 \
    --num_classes 100 \
    --model_name classifier_cifar100_test \
    

# Sample
: '
python sample_cifar.py \
    --image_size 32 \
    --diffusion_steps 1000 \
    --num_res_blocks 3 \
    --noise_schedule linear \
    --batch_size 10 \
    --num_samples 10 \
    --learn_sigma True \
    --class_cond True \
    --num_channels 128 \
    --classfier_scale 1 \
    --out_path ../images/apple \
    --classifier_path ../log/classifier_cifar100/model004999.pt \
    --model_path ../log/model005000.pt
'
