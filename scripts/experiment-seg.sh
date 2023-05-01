ids=$1

mem=attention

for mem in attention none; do
    for seg in 2 4 8 16; do
        if [[ $seg = 16 ]]; then
            bs=16
            g_check=True
        else
            bs=32
            g_check=False
        fi
        for wpos in right; do
            sh train.sh $ids $mem $seg $wpos $bs $g_check
        done
    done
done
