ids=$1

# mem=attention
wpos=left

# residual means sth different
for mem in residual; do
for seg in 6; do
    if (( $seg > 7 )); then
        bs=32
        g_check=False
    else
        bs=64
        g_check=False
    fi
    WANDB__SERVICE_WAIT=300 sh train_small.sh $ids $mem $seg $wpos $bs $g_check $seg 5 outputs/small/blenderbot-400M-RMT-seg4-mem_rleft_wleft_5_residual-continual-prev_sess/checkpoint-2000/pytorch_model.bin prev_sess
done
done
