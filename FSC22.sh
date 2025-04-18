#!/bin/bash

#train
python3 train.py \
			--dataset="FSC22" \
			--fold=$1 \
			--mode='validation' \
			--optim_type="adam" \
			--lr=1e-4 \
			--wd=1e-4 \
			--lr_decay=0.97 \
			--n_epoch=30 \
			--batch_size=1 \
			--folder=$2 \
 			--device=$3