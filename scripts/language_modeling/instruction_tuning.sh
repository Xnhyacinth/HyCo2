## mistral-7b + sfr
random_port=$((RANDOM%(65535-1024+1)+1024))
while [[ $(ss -tln | grep ":$random_port") ]]; do
    random_port=$((RANDOM%(65535-1024+1)+1024))
done
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_port ${random_port} \
    -m src.language_modeling.train \
        --config config/language_modeling/finetune.yaml \
        --chat_format mistral \
        # --max_train_samples 1000
        # --train_file data/instruction_tuning/processed/ablation_data.jsonl
    


## mixtral-moe + sfr
# accelerate launch \
#     --config_file accelerate_fsdp.config \
#     -m src.language_modeling.train \
#         --config config/language_modeling/finetune.yaml \
#         --chat_format mixtral --model_name_or_path wandb/run-20240310_094951-li520mhm/files/checkpoint/last \
#         --exp_name mixtral_moe \
#         --per_device_train_batch_size 1 --gradient_accumulation_steps 8