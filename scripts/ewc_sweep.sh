#!/bin/bash

NAME="${1:?'Name must be set'}"
OUT="$SCRATCH/slurm_out/$NAME/%x-%A_%a.out"
N_JOBS="${2:?'N_JOBS must be set'}"
ARGS="${@:3}"
#ARGS="--multihead --unsupervised_epochs_per_task 0 \
#     --supervised_epochs_per_task 50 --no_wandb_cleanup  \
#     --tags cifar100 debugging ewc resnet18 --run_group ewc_sweep
#     --encoder_model resnet18 --pretrained \
#     --dataset cifar100 --n_classes_per_task 20 \
#     "

echo "Sweep with name '$NAME' and with args '$ARGS'"
echo "Number of jobs per task: $N_JOBS"
# Create the slurm output dir if it doesn't exist already.

# activate the virtual environment (only used to download the datasets)
source ~/ENV/bin/activate
python -m scripts.download_datasets --data_dir "$SCRATCH/data"
python -m scripts.download_pretrained_models # --save_dir "$SCRATCH/checkpoints"
deactivate

zip -u "$SCRATCH/data.zip" "$SCRATCH/data"

mkdir -p "$SCRATCH/slurm_out/$NAME"


sbatch --output $OUT --job-name baseline   --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "baseline"   $ARGS
sbatch --output $OUT --job-name ewc_01     --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "ewc_01"     $ARGS --ewc.coef 0.1
sbatch --output $OUT --job-name ewc_1      --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "ewc_1"      $ARGS --ewc.coef 1
sbatch --output $OUT --job-name ewc_10     --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "ewc_10"     $ARGS --ewc.coef 10
sbatch --output $OUT --job-name ewc_100    --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "ewc_100"    $ARGS --ewc.coef 100
sbatch --output $OUT --job-name ewc_1000   --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "ewc_1000"   $ARGS --ewc.coef 1000
sbatch --output $OUT --job-name ewc_10000  --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "ewc_10000"  $ARGS --ewc.coef 10000
sbatch --output $OUT --job-name ewc_100000 --time 12:00:00 --array=1-$N_JOBS ./scripts/run.sh --run_name "ewc_100000" $ARGS --ewc.coef 100000
