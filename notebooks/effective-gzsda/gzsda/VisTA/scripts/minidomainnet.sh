#!/bin/bash

cd ..

# custom config
DATA= # you may change your path to dataset here
TRAINER=VISTA

DATASET=minidomainnet # name of the dataset
CFG=minidomainnet  # config file
SEED=2025
GPU=0,1

NAME=cp
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains painting --seed ${SEED}  

NAME=cr
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains real --seed ${SEED}  

NAME=cs
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains clipart --target-domains sketch --seed ${SEED}  

NAME=pc
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA}  --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains clipart --seed ${SEED}  

NAME=pr
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA}  --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains real --seed ${SEED}  

NAME=ps
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA}  --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains painting --target-domains sketch --seed ${SEED}  

NAME=rc
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA}  --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains clipart --seed ${SEED}  

NAME=rp
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains painting --seed ${SEED}  

NAME=rs
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains real --target-domains sketch --seed ${SEED}  

NAME=sc
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains clipart --seed ${SEED}  

NAME=sp
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains painting --seed ${SEED}  

NAME=sr
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains sketch --target-domains real --seed ${SEED}  