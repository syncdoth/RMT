ids=$1

# mem=attention
wpos=left


for mem in residual; do
for seg in 8; do
    if (( $seg > 6 )); then
        bs=4
        g_check=True
    else
        bs=4
        g_check=False
    fi
    WANDB__SERVICE_WAIT=300 sh train.sh $ids $mem $seg $wpos $bs $g_check $seg 5
done
done
