#!/bin/bash

# Usage:
# ./run_single_job.sh MODEL DATASET AUG PIPE SEED
#
# Example:
# ./run_single_job.sh wideresnet_34_10 cifar10 basic at 0

MODEL=$1
DATASET=$2
AUG=$3
PIPE=$4
SEED=$5

if [[ -z "$MODEL" || -z "$DATASET" || -z "$AUG" || -z "$PIPE" || -z "$SEED" ]]; then
    echo "Usage: $0 MODEL DATASET AUG PIPE SEED"
    exit 1
fi

OUTPUT_DIR="./"
mkdir -p ${OUTPUT_DIR}

########################################
# Dataset dependent parameters
########################################

if [[ "$DATASET" == "cifar10" ]]; then
    NUM_CLASSES=10
    BETA=20
else
    NUM_CLASSES=100
    BETA=150
fi

CORR_DATASET="${DATASET}-c"

########################################
# Pipeline specific settings
########################################

MODEL_NAME="$MODEL"
TRAIN_CONFIG="train_config_${PIPE}_l2"
EVAL_CONFIG="eval_base"
EXPERIMENT_NAME="${DATASET}_${PIPE}_${AUG}_l2"

if [[ "$PIPE" == "emff" ]]; then
    MODEL_NAME="${MODEL}_emff"
    EVAL_CONFIG="eval_emff"
fi

if [[ "$PIPE" == "evat" ]]; then
    EVAL_CONFIG="eval_evidential"
fi

########################################
# Job naming
########################################
JOB_NAME="${MODEL_NAME}_${DATASET}_${PIPE}_${AUG}_s${SEED}_l2"


FILE_PATH="${OUTPUT_DIR}/${JOB_NAME}.sh"

########################################
# Training command
########################################

if [[ "$PIPE" == "evat" ]]; then

TRAIN_CMD="python src/train.py --config-name ${TRAIN_CONFIG} \\
        dataloaders.aug=${AUG} \\
        experiment.dataset=${DATASET} \\
        experiment.num_classes=${NUM_CLASSES} \\
        experiment.seed=${SEED} \\
        experiment.beta=${BETA} \\
        experiment.max_reg_factor=0.1 \\
        experiment.reg_scheduler_gamma=0.002 \\
        loss=expected_nll_loss \\
        model=${MODEL} \\
        proxy=${MODEL}_proxy \\
        experiment.name=${EXPERIMENT_NAME} \\
        step=fit"

else

TRAIN_CMD="python src/train.py --config-name ${TRAIN_CONFIG} \\
        dataloaders.aug=${AUG} \\
        experiment.dataset=${DATASET} \\
        experiment.num_classes=${NUM_CLASSES} \\
        experiment.seed=${SEED} \\
        model=${MODEL_NAME} \\
        experiment.name=${EXPERIMENT_NAME} \\
        step=fit"

if [[ "$PIPE" == *"awp"* || "$PIPE" == *"ikl"* ]]; then
TRAIN_CMD="${TRAIN_CMD} \\
        proxy=${MODEL_NAME}_proxy"
fi

fi

########################################
# Generate job script
########################################

cat > ${FILE_PATH} <<EOF
#!/bin/bash
cd ../

echo "--------------- Running the code ---------------"
echo "Job: ${JOB_NAME}"
date

# -------------------------
# Training
# -------------------------
${TRAIN_CMD}

# -------------------------
# Evaluation
# -------------------------

# AA
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=autoattack_l2 \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

# PGD100
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_05_100_l2 \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

# PGD20
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_05_20_l2 \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

# Corruptions
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${CORR_DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    ~attacks \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

echo "Completed at:"
date
EOF

chmod +x ${FILE_PATH}

########################################
# Run job immediately
########################################

echo "Launching job ${JOB_NAME}"
sh ${FILE_PATH}