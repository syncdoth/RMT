ids=$1

mem=attention

for seg in 2 4 8; do
sh train.sh $ids $mem $seg
done
