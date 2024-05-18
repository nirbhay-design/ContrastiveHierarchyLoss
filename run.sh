#!/bin/bash

NCCL_SHM_DISABLE=1 nohup python train.py configs/LCAWContrastiveLoss_exp1.yaml > logs/TieredImagenet_r18.log &