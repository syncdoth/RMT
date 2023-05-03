# evaluate different segment length at eval
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

function evaluate(){
    train_seg=$1
    eval_seg=$2
    ckpt_path=$3
    len=$4
    gate=$5
    script="$launcher main.py \
        --model_name 'facebook/blenderbot-3B' \
        --wandb_run_name blenderbot3B-eval-train${train_seg}-eval${eval_seg}-mem_${len}_${gate} \
        --eval_accumulation_steps 100 \
        --report_to 'wandb' \
        --save_strategy 'steps' \
        --per_device_eval_batch_size 64 \
        --test_max_session 5 \
        --use_lora --bf16 \
        --test_only \
        --load_checkpoint $ckpt_path \
        --eval_num_segments $eval_seg \
        --memory_length $len \
        --memory_position left --write_memory_position right \
        --memory_gate_type $gate"

    eval $script
}

memlen=10

for memgate in none attention; do
    for train_seg in 2 4 8 16; do
        for eval_seg in 2 4 8 16 -1; do
            WANDB__SERVICE_WAIT=300 evaluate $train_seg $eval_seg outputs/blenderbot3B-RMT-seg$train_seg-mem_rleft_wright_${memlen}_$memgate/checkpoint-1000/pytorch_model.bin $memlen $memgate
        done
    done
done
