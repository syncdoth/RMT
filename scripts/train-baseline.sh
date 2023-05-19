##### stdin ####
ids=$1
################
# compute number of gpus
arrIDs=(${ids//,/ })
GPU_PER_NODE="${#arrIDs[@]}"

# decide python launcher
if [ $GPU_PER_NODE = 1 ]; then
    echo "Using 1 GPU: use simple python launcher..."
    launcher="CUDA_VISIBLE_DEVICES=$ids python"
else
    echo "Using multi-GPU: using torchrun launcher..."
    launcher="CUDA_VISIBLE_DEVICES=$ids WORLD_SIZE=$GPU_PER_NODE torchrun --nproc_per_node $GPU_PER_NODE"
fi

# hyperparameters defined here
lr=1e-6
max_steps=1000

# define script
script="$launcher main.py \
    --model_name facebook/blenderbot-3B \
    --wandb_run_name blenderbot3B-msc-textnorm \
    --learning_rate $lr \
    --warmup_steps 0 \
    --weight_decay 0 \
    --evaluation_strategy no \
    --eval_steps 500 \
    --eval_accumulation_steps 100 \
    --max_steps $max_steps \
    --num_train_epochs 5 \
    --report_to 'wandb' \
    --output_dir outputs/blenderbot3B-msc \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --save_steps $max_steps \
    --save_total_limit 3 \
    --load_best_model_at_end False \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing False \
    --num_segments 1 \
    --eval_num_segments 1 \
    --train_max_session 4 \
    --valid_max_session 5 \
    --test_max_session 5 \
    --memory_length 0 \
    --use_lora --bf16"

# run the script
eval $script
