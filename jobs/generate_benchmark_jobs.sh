#!/bin/bash

MODEL=$1
DATASET=$2
shift 2

# If user provides seeds -> use them
if [ "$#" -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(0 1 2)
fi

if [[ -z "$MODEL" || -z "$DATASET" ]]; then
    echo "Usage: $0 [model] [dataset] [seeds...]"
    echo "Example:"
    echo "  $0 wideresnet_34_10 cifar10"
    echo "  $0 wideresnet_34_10 cifar10 0 1 2 3 4"
    exit 1
fi

OUTPUT_DIR="benchmark_${MODEL}_${DATASET}"
mkdir -p ${OUTPUT_DIR}

AUGS=("basic" "cutout" "augmix" "autoaug")
BASE_PIPES=("at" "at_awp" "trades" "trades_awp" "ikl" "emff")
OURS_PIPES=("evat")

CORR_DATASET="${DATASET}-c"

if [[ "$DATASET" == "cifar10" ]]; then
    NUM_CLASSES=10
    BETA=20
else
    NUM_CLASSES=100
    BETA=150
fi

############################################
# BASELINE JOBS
############################################

for AUG in "${AUGS[@]}"; do
for PIPE in "${BASE_PIPES[@]}"; do
for SEED in "${SEEDS[@]}"; do

MODEL_NAME="$MODEL"
if [[ "$PIPE" == "emff" ]]; then
    MODEL_NAME="${MODEL}_emff"
fi

JOB_NAME="${MODEL_NAME}_${DATASET}_${PIPE}_${AUG}_s${SEED}"
EXPERIMENT_NAME="${DATASET}_${PIPE}_${AUG}"
TRAIN_CONFIG="train_config_${PIPE}"

if [[ "$PIPE" == "emff" ]]; then
    EVAL_CONFIG="eval_emff"
else
    EVAL_CONFIG="eval_base"
fi

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

FILE_PATH="${OUTPUT_DIR}/${JOB_NAME}.sh"

cat > ${FILE_PATH} <<EOF
#!/bin/bash
cd ../../

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
    attacks=autoattack \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

# PGD100
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_8_100 \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

# PGD20 - eps 1,2,4,6,8
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_1_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_2_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_4_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_6_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL_NAME} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_8_20 \\
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

done
done
done

############################################
# OUR METHOD
############################################

for AUG in "${AUGS[@]}"; do
for SEED in "${SEEDS[@]}"; do

PIPE="evat"

JOB_NAME="${MODEL}_${DATASET}_${PIPE}_${AUG}_s${SEED}"
EXPERIMENT_NAME="${DATASET}_${PIPE}_${AUG}"
TRAIN_CONFIG="train_config_${PIPE}"
EVAL_CONFIG="eval_evidential"

FILE_PATH="${OUTPUT_DIR}/${JOB_NAME}.sh"

cat > ${FILE_PATH} <<EOF
#!/bin/bash
cd ../../

echo "--------------- Running the code ---------------"
echo "Job: ${JOB_NAME}"
date

# -------------------------
# Training
# -------------------------
python src/train.py --config-name ${TRAIN_CONFIG} \\
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
        step=fit

# -------------------------
# Evaluation
# -------------------------

# AA
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=autoattack \\
    experiment.seed=${SEED} \\
    model=${MODEL} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

# PGD100
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_8_100 \\
    experiment.seed=${SEED} \\
    model=${MODEL} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

# PGD20 - eps 1,2,4,6,8
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_1_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_2_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_4_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_6_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    attacks=pgd_8_20 \\
    experiment.seed=${SEED} \\
    model=${MODEL} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

# Corruptions
python src/eval.py --config-name ${EVAL_CONFIG} \\
    ckpt_choice=best \\
    experiment.dataset=${CORR_DATASET} \\
    experiment.num_classes=${NUM_CLASSES} \\
    ~attacks \\
    experiment.seed=${SEED} \\
    model=${MODEL} \\
    experiment.name=${EXPERIMENT_NAME} \\
    step=test

echo "Completed at:"
date
EOF

chmod +x ${FILE_PATH}

done
done

############################################
# RUN ALL SCRIPT
############################################

cat > ${OUTPUT_DIR}/run_all.sh <<EOF
#!/bin/bash

for job in *.sh
do
    if [[ "\$job" != "run_all.sh" ]]; then
        sh \$job
    fi
done
EOF

chmod +x ${OUTPUT_DIR}/run_all.sh

echo "All jobs generated in ${OUTPUT_DIR}"