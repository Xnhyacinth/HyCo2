
bash src/model/install.sh
pip install -U accelerate


bash scripts/language_modeling/gate_pt.sh gate_ck > logs/pretrain/gated_1e-4_pos_fproj.log 2>&1 
sleep 200

bash scripts/language_modeling/gate_pt.sh gate_ck1 > logs/pretrain/gated_1e-4_pos_fproj_ptuning_mlp2x_gelu_selectp_calp_chunk_red_mypt_stage2.log 2>&1 
sleep 200

bash scripts/language_modeling/gate.sh gate_ck0 > logs/finetune/gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10.log 2>&1 

# sleep 200
bash scripts/eval.sh 6 gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10 32 contriever 3 gated nq mistral &
bash scripts/eval.sh 7 gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10 32 contriever 3 gated tqa mistral &
sleep 30
bash scripts/eval.sh 0 gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10 32 contriever 3 gated hotpotqa mistral &
bash scripts/eval.sh 1 gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10 32 contriever 5 gated nq mistral &
sleep 30
bash scripts/eval.sh 2 gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10 32 contriever 5 gated tqa mistral &
bash scripts/eval.sh 3 gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10 32 contriever 5 gated hotpotqa mistral &
sleep 40
bash scripts/eval.sh 4 gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10 32 contriever 3 gated 2wikimqa mistral &
bash scripts/eval.sh 5 gated_pt1e-4_lr2e-5_pos_fproj_ptuning_mlp2x_gelu_stage3_e1_m4096_r2048_top10 32 contriever 3 gated wq mistral &

wait