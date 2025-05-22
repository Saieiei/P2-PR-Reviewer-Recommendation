#!/usr/bin/env bash
# sweep_hparams_graphcodebert.sh
# Hyperparameter sweep for GraphCodeBERT with progress updates

export OMP_NUM_THREADS=128
export MKL_NUM_THREADS=128

# detect GPUs
readarray -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "âŒ No GPUs found!"
  exit 1
fi
GPU_ID=${GPUS[0]}

# hyperparam grid
LRS=(1e-5 2e-5 5e-5)
BSS=(16 32 64)
EPOCHS=(2 3 5)

DATA_DIR="."
MODEL_TYPE="roberta"
MODEL_NAME_OR_PATH="./models/graphcodebert-base"
TOKENIZER_NAME="./models/graphcodebert-base"
TASK_NAME="codesearch"
MAX_SEQ_LENGTH=512
TRAIN_FILE="train.tsv"
DEV_FILE="dev.tsv"
OUTPUT_ROOT="./sweep_results/graphcodebert"
mkdir -p "$OUTPUT_ROOT"

TOTAL_JOBS=$(( ${#LRS[@]} * ${#BSS[@]} * ${#EPOCHS[@]} ))
BAR_LEN=40
job=0

for LR in "${LRS[@]}"; do
  for BS in "${BSS[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
      ((job++))
      EXP_NAME="lr${LR}_bs${BS}_ep${EPOCH}"
      OUTPUT_DIR="$OUTPUT_ROOT/$EXP_NAME"
      mkdir -p "$OUTPUT_DIR"

      if [[ -f "$OUTPUT_DIR/eval_results.txt" ]]; then
        echo "í ½í¿¡ Skipping $EXP_NAME (already done)"
        continue
      fi

      percent=$(( job * 100 / TOTAL_JOBS ))
      filled=$(( percent * BAR_LEN / 100 ))
      empty=$(( BAR_LEN - filled ))
      bar=$(printf "%${filled}s" | tr ' ' '#')$(printf "%${empty}s")
      echo "[$job/$TOTAL_JOBS] [$bar] ${percent}% â†’ Running $EXP_NAME on GPU $GPU_ID"

      CUDA_VISIBLE_DEVICES=$GPU_ID python run_classifier.py \
        --data_dir "$DATA_DIR" \
        --model_type "$MODEL_TYPE" \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --tokenizer_name "$TOKENIZER_NAME" \
        --task_name "$TASK_NAME" \
        --train_file "$TRAIN_FILE" \
        --dev_file "$DEV_FILE" \
        --max_seq_length $MAX_SEQ_LENGTH \
        --per_gpu_train_batch_size $BS \
        --per_gpu_eval_batch_size $BS \
        --learning_rate $LR \
        --num_train_epochs $EPOCH \
        --dataloader_num_workers 64 \
        --output_dir "$OUTPUT_DIR" \
        --do_train \
        --do_eval \
        --fp16
    done
  done
done

echo "âœ… Hyperparameter sweep complete ($TOTAL_JOBS jobs)."
