##### stdin ####
ids=$1
memory_gate=$2
num_seg=$3
write_memory_position=$4
train_bs=$5
grad_check=$6
eval_seg=$7
t_session=$8
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
max_steps=2000
memory_length=5
memory_position=left

run_name=blenderbot3B-RMT-seg$num_seg-mem_r${memory_position}_w${write_memory_position}_${memory_length}_$memory_gate-textnormv3-proj-scaledv2

script="$launcher main.py \
    --model_name facebook/blenderbot-3B \
    --wandb_run_name $run_name \
    --learning_rate $lr \
    --warmup_steps 0 \
    --weight_decay 0 \
    --eval_steps 500 \
    --eval_accumulation_steps 100 \
    --max_steps $max_steps \
    --num_train_epochs 5 \
    --report_to 'wandb' \
    --output_dir outputs/big/$run_name \
    --logging_steps 10 \
    --save_strategy 'steps' \
    --save_steps 500 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --per_device_train_batch_size $train_bs \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing $grad_check \
    --num_segments $num_seg \
    --eval_num_segments $eval_seg \
    --train_max_session 4 \
    --valid_max_session 5 \
    --test_max_session 5 \
    --test_target_session $t_session \
    --memory_length $memory_length \
    --memory_position $memory_position --write_memory_position $write_memory_position \
    --memory_gate_type $memory_gate --task remember_sess1 \
    --use_lora --bf16"

eval $script
