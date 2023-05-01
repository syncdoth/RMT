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
    script="$launcher main.py \
        --model_name 'facebook/blenderbot-3B' \
        --wandb_run_name blenderbot3B-eval-train${train_seg}-eval${eval_seg} \
        --eval_accumulation_steps 100 \
        --report_to 'wandb' \
        --save_strategy 'steps' \
        --per_device_eval_batch_size 32 \
        --test_max_session 5 \
        --bf16 \
        --test_only \
        --load_checkpoint $ckpt_path/pytorch_model.bin \
        --eval_num_segments $eval_seg"

    eval $script
}

memlen=10

for memgate in none attention; do
    for train_seg in 2 4 8 16; do
        for eval_seg in 2 4 8 16 -1; do
            evaluate $train_seg $eval_seg outputs/blenderbot3B-RMT-seg$train_seg-mem_${memlen}_${memgate}/checkpoint-1000
        done
    done
done