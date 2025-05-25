

config=${1:-"cformer_pt"}
gpu=${2:-"8"}
max_train_samples=${3:-"100000000"}
random_port=$((RANDOM%(65535-1024+1)+1024))
while [[ $(ss -tln | grep ":$random_port") ]]; do
    random_port=$((RANDOM%(65535-1024+1)+1024))
done
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes ${gpu} \
    --main_process_port ${random_port} \
    -m src.language_modeling.train \
        --config config/language_modeling/${config}.yaml \
