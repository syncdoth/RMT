# evaluate different segment length at eval
##### stdin ####
ids=$1
################

function evaluate(){
    eval_seg=$1
    ckpt_path=$2
    len=$3
    gate=$4
    out=$5
    wpos=$6
    size=$7
    script="CUDA_VISIBLE_DEVICES=$ids python infer.py \
        --model_name 'facebook/blenderbot-$size' \
        --per_device_eval_batch_size 16 \
        --test_max_session 5 \
        --test_target_session 200 \
        --task prev_sess \
        --bf16 \
        --load_checkpoint $ckpt_path \
        --eval_num_segments $eval_seg \
        --memory_length $len \
        --memory_position left --write_memory_position $wpos \
        --memory_gate_type $gate \
        --out_file $out"

    eval $script
}

memlen=10
train_seg=16
eval_seg=-1
memgate=none
wpos=left
size=400M-distill

# baseline
model_name=blenderbot400M-msc
evaluate 1 outputs/small/$model_name/checkpoint-2000/pytorch_model.bin 0 none generated/final/$model_name.jsonl left $size
# seg1
model_name=blenderbot-400M-RMT-seg1-mem_rleft_wleft_5_residual-textnormv3-proj-scaledv2
evaluate 1 outputs/small/$model_name/checkpoint-2000/pytorch_model.bin 5 residual generated/final/$model_name.jsonl left $size
seg2
model_name=blenderbot-400M-RMT-seg2-mem_rleft_wleft_5_residual-continual-prev_sess
evaluate 2 outputs/small/$model_name/checkpoint-2000/pytorch_model.bin 5 residual generated/final/$model_name.jsonl left $size
# seg4
model_name=blenderbot-400M-RMT-seg4-mem_rleft_wleft_5_residual-continual-prev_sess
evaluate 4 outputs/small/$model_name/checkpoint-2000/pytorch_model.bin 5 residual generated/final/$model_name.jsonl left $size
# seg8
model_name=blenderbot-400M-RMT-seg8-mem_rleft_wleft_5_residual-continual-prev_sess
evaluate 8 outputs/small/$model_name/checkpoint-2000/pytorch_model.bin 5 residual generated/final/$model_name.jsonl left $size
