# evaluate different segment length at eval
##### stdin ####
ids=$1
################

function evaluate(){
    eval_seg=$1
    ckpt_path=$2
    len=$3
    gate=$4
    script="CUDA_VISIBLE_DEVICES=$ids python infer.py \
        --model_name 'facebook/blenderbot-3B' \
        --per_device_eval_batch_size 16 \
        --test_max_session 5 \
        --bf16 \
        --load_checkpoint $ckpt_path \
        --eval_num_segments $eval_seg \
        --memory_length $len \
        --memory_position left --write_memory_position right \
        --memory_gate_type $gate"

    eval $script
}

evaluate -1 outputs/blenderbot3B-RMT-seg16-mem_rleft_wright_10_none/checkpoint-1000/pytorch_model.bin 10 none