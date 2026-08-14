#!/bin/bash

cd ..

# custom config
DATA= # you may change your path to dataset here
TRAINER=VISTA

DATASET=officehome # name of the dataset
CFG=officehome  # config file
SEED=2025
GPU=0,1

NAME=ac
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains clipart --seed ${SEED}

NAME=pc
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains clipart --seed ${SEED}

NAME=rc
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains clipart --seed ${SEED}

NAME=ap
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains product --seed ${SEED}

NAME=ar
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains art --target-domains real_world --seed ${SEED}

NAME=ca
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains art --seed ${SEED}

NAME=cp
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains product --seed ${SEED}

NAME=cr
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains real_world --seed ${SEED}

NAME=pa
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains art --seed ${SEED}

NAME=pr
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains product --target-domains real_world --seed ${SEED}

NAME=ra
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains art --seed ${SEED}

NAME=rp
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real_world --target-domains product --seed ${SEED}