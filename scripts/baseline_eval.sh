ids=$1
GPU_PER_NODE=$2

# CUDA_VISIBLE_DEVICES=$ids python main.py \
CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$GPU_PER_NODE torchrun --nproc_per_node $GPU_PER_NODE main.py \
    --model_name 'facebook/blenderbot-3B' \
    --wandb_run_name "blenderbot3B-baseline-test-correct" \
    --eval_accumulation_steps 50 \
    --report_to 'wandb' \
    --output_dir 'outputs' \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --per_device_eval_batch_size 16 \
    --test_max_session 5 \
    --bf16 \
    --test_only --load_baseline

CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$GPU_PER_NODE torchrun --nproc_per_node $GPU_PER_NODE main.py \
    --model_name 'facebook/blenderbot-3B' \
    --wandb_run_name "blenderbot3B-baseline-valid-correct" \
    --eval_accumulation_steps 50 \
    --report_to 'wandb' \
    --output_dir 'outputs' \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --per_device_eval_batch_size 16 \
    --test_max_session 5 \
    --bf16 \
    --test_only --load_baseline \
    --test_data_path msc/session_5/valid.txt