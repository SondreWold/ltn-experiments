for task in 'QNLI' 'WNLI' 'RTE' 'SST' 'COLA' 'MNLI'
do
    python3 train.py --task=$task --epochs=1
    python3 train.py --logic_mode --task=$task --epochs=1
done
