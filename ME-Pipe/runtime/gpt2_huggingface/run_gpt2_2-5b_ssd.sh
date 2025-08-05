#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="../../results"
MODEL="gpt2_2-5b"
CONFIG="../../../model_lib/gpt2_configs/gpt2-2-5-billion.json"
OUT_DIR="./logs/${MODEL}"
mkdir -p ${OUT_DIR}
# --------------------------------------------
# for SCHEDULED in "D16_vDP_N4_Top1" "D16_vPP_N4_Top1" "D64_vDP_N4_Top1" "D64_vPP_N4_Top1" "D256_vDP_N4_Top1" "D256_vPP_N4_Top1"
for SCHEDULED in D128_vPP_N4_W2_new_Top1 # D32_vPP_N4_Ufwd1_Ubwd1_P2 # "D32_vPP_N4_Top1_gpt2_10b_3090"
do
# echo "Clean Python Processes"
# sleep 3s && pkill -9 python3 && pkill -9 python && sleep 3s

echo "${SCHEDULED}"
# numactl --cpunodebind=0 --membind=0 \
python3 main_ssd.py \
--gpt2_train_file "../../../data/wikitext-103-tokens/wiki.train.tokens" \
--gpt2_config_path ${CONFIG} \
--gpt2_model "" \
--module_dir ${MODEL_DIR} \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_iter 4 \
--gl_window_size 2 \
--output_dir ${OUT_DIR} \
--suffix "_gpt2_2-5b_3090"
# |& tee output.txt
# |& tee ${OUT_DIR}/${SCHEDULED}__fig10_fig12.txt
done
