ids=$1
# GPU_PER_NODE=$2

# CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$GPU_PER_NODE torchrun --nproc_per_node $GPU_PER_NODE \
CUDA_VISIBLE_DEVICES=$ids python main.py \
    --model_name 'facebook/blenderbot-400M-distill' \
    --wandb_run_name 'blenderbot-RMT-seg3-train_session4-test_session5' \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --weight_decay 0 \
    --eval_steps 1000 \
    --max_steps -1 \
    --num_train_epochs 5 \
    --report_to 'wandb' \
    --output_dir 'outputs' \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --save_steps 1000 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing False \
    --num_segments 3 \
    --train_max_session 4 \
    --valid_max_session 5 \
    --test_max_session 5 \
    --memory_length 10 \
    --memory_position left --write_memory_position right

    # --use_lora --fp16 --train_8bit
