#!/usr/bin/env bash
# coding=utf-8
#
# hyperparam sweep for a single A100 GPU,
# using all 128 CPU threads for tokenization & feature I/O

# —— Only one concurrent GPU job ——
MAX_JOBS=1

# —— Use all your CPU threads for OMP/MKL ——
export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128

# —— discover NVIDIA GPUs (you’ll see only 1 A100) ——
mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
NUM_GPUS=${#GPUS[@]}
if (( NUM_GPUS == 0 )); then
  echo "❌ No NVIDIA GPUs found!"
  exit 1
fi

# —— fixed args ——  
DATA_DIR="."
MODEL_TYPE="roberta"
MODEL_NAME_OR_PATH="./models/codebert-base"
TOKENIZER_NAME="./models/codebert-base"
TASK_NAME="codesearch"
MAX_SEQ_LENGTH=200
TRAIN_FILE="train.tsv"
DEV_FILE="dev.tsv"
OUTPUT_ROOT="./sweep_results"
mkdir -p "${OUTPUT_ROOT}"

job=0
for LR in 1e-5 2e-5 5e-5; do
  for BS in 16 32 64; do
    for EPOCHS in 2 3 5; do

      EXP_NAME="lr${LR}_bs${BS}_ep${EPOCHS}"
      OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
      EVAL_FILE="${OUTPUT_DIR}/eval_results.txt"

      # skip if already done
      if [[ -f "${EVAL_FILE}" ]]; then
        echo "⏭️  Skipping ${EXP_NAME}"
        continue
      fi

      # throttle: wait every MAX_JOBS launches
      if (( job > 0 && job % MAX_JOBS == 0 )); then
        wait
      fi

      # only GPU 0 in your system
      GPU_ID=${GPUS[0]}

      (
        echo "Running ${EXP_NAME} on GPU ${GPU_ID}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python run_classifier.py \
          --data_dir ${DATA_DIR} \
          --model_type ${MODEL_TYPE} \
          --model_name_or_path ${MODEL_NAME_OR_PATH} \
          --tokenizer_name ${TOKENIZER_NAME} \
          --task_name ${TASK_NAME} \
          --train_file ${TRAIN_FILE} \
          --dev_file ${DEV_FILE} \
          --max_seq_length ${MAX_SEQ_LENGTH} \
          --per_gpu_train_batch_size ${BS} \
          --per_gpu_eval_batch_size ${BS} \
          --learning_rate ${LR} \
          --num_train_epochs ${EPOCHS} \
          --dataloader_num_workers 64 \
          --output_dir ${OUTPUT_DIR} \
          --do_train \
          --do_eval \
          --fp16
      ) &

      ((job++))

    done
  done
done

# wait for the final job
wait
echo "✅ All sweeps complete."
