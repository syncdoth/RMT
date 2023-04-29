ids=$1
# GPU_PER_NODE=$2
memory_gate=$2
num_seg=$3

# hyperparameters defined here
lr=1e-5
max_steps=2000
memory_length=10

# For multi-gpu, uncomment below and comment-out python main.py
# CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$GPU_PER_NODE torchrun --nproc_per_node $GPU_PER_NODE main.py \
CUDA_VISIBLE_DEVICES=$ids python main.py \
    --model_name 'facebook/blenderbot-3B' \
    --wandb_run_name "blenderbot3B-RMT-seg$num_seg-train_session4-test_session5-$memory_gate" \
    --learning_rate $lr \
    --warmup_steps 0 \
    --weight_decay 0 \
    --eval_steps 100 \
    --eval_accumulation_steps 100 \
    --max_steps $max_steps \
    --num_train_epochs 5 \
    --report_to 'wandb' \
    --output_dir 'outputs' \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing False \
    --num_segments $num_seg \
    --train_max_session 4 \
    --valid_max_session 5 \
    --test_max_session 5 \
    --memory_length 10 \
    --memory_position left --write_memory_position right \
    --memory_gate_type $memory_gate \
    --use_lora --bf16
