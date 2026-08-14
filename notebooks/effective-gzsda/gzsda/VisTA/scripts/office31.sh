#!/bin/bash

cd ..

# custom config
DATA=/opt/data/private # you may change your path to dataset here
TRAINER=VISTA

DATASET=office31 # name of the dataset
CFG=office31  # config file
SEED=2025
GPU=0,1

NAME=ad
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains amazon --target-domains dslr --seed ${SEED} 

NAME=aw
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains amazon --target-domains webcam --seed ${SEED} 

NAME=da
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains dslr --target-domains amazon --seed ${SEED} 

NAME=dw
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains dslr --target-domains webcam --seed ${SEED} 

NAME=wa
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains webcam --target-domains amazon --seed ${SEED} 

NAME=wd
DIR=output/${DATASET}/${TRAINER}/${NAME}/seed_${SEED}
CUDA_VISIBLE_DEVICES=${GPU} python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains webcam --target-domains dslr --seed ${SEED} 