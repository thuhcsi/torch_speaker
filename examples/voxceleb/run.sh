#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2
voxceleb1_path=/home/zrs/datasets/VoxCeleb/voxceleb1
voxceleb2_path=/home/zrs/datasets/VoxCeleb/voxceleb2

stage=2

if [ $stage -eq 0 ];then
    rm -rf data/train/
    mkdir -p data/train/

    ln -s ${voxceleb1_path}/vox1_dev_wav/* data/train/
    ln -s ${voxceleb2_path}/dev/aac/* data/train
fi

if [ $stage -eq 1 ];then
    echo build train data list
    python3 scripts/build_datalist.py \
	    --extension wav \
	    --dataset_dir data/train \
	    --data_list_path data/train.csv

    echo format trial list
    python3 scripts/format_trials.py \
		--voxceleb1_root $voxceleb1_path \
		--src_trials_path $voxceleb2_path/List_of_trial_pairs-VoxCeleb1-Clean.txt \
		--dst_trials_path data/vox1_clean.txt

    python3 scripts/format_trials.py \
		--voxceleb1_root $voxceleb1_path \
		--src_trials_path $voxceleb2_path/List_of_trial_pairs-VoxCeleb1-H-Clean.txt \
		--dst_trials_path data/vox1_H_clean.txt

    python3 scripts/format_trials.py \
		--voxceleb1_root $voxceleb1_path \
		--src_trials_path $voxceleb2_path/List_of_trial_pairs-VoxCeleb1-E-Clean.txt \
		--dst_trials_path data/vox1_E_clean.txt
fi

if [ $stage -eq 2 ];then
    python3 tools/train.py \
		--config config/voxceleb.yaml
fi

if [ $stage -eq 3 ];then
    python3 tools/evaluate.py \
			--config config/voxceleb.yaml \
			--trial_path data/vox1_E_clean.txt \
			--checkpoint_path ckpt.pt

    python3 tools/evaluate.py \
			--config config/voxceleb.yaml \
			--trial_path data/vox1_H_clean.txt \
			--checkpoint_path ckpt.pt
fi

