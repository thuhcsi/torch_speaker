#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
voxceleb1_path=~/datasets/VoxCeleb/voxceleb1
voxceleb2_path=~/datasets/VoxCeleb/voxceleb2
musan_path=~/datasets/musan/

stage=2

if [ $stage -eq 0 ];then
	rm -rf data/train/
	rm -rf data/noise/
	mkdir -p data/train/
	mkdir -p data/noise/

	ln -s ${voxceleb1_path}/vox1_dev_wav/* data/train/
	ln -s ${voxceleb2_path}/dev/aac/* data/train
	ln -s $musan_path/* data/noise
fi

if [ $stage -eq 1 ];then
	echo build train data list
	python3 scripts/build_datalist.py \
		--extension wav \
		--dataset_dir data/train \
		--data_list_path data/train.csv

	python3 scripts/build_datalist.py \
		--extension wav \
		--dataset_dir data/noise \
		--data_list_path data/noise.csv

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
		--trial_path data/vox1_clean.txt \
		--checkpoint_path ckpt.pt

	python3 tools/evaluate.py \
		--config config/voxceleb.yaml \
		--trial_path data/vox1_E_clean.txt \
		--checkpoint_path ckpt.pt

	python3 tools/evaluate.py \
		--config config/voxceleb.yaml \
		--trial_path data/vox1_H_clean.txt \
		--checkpoint_path ckpt.pt
fi

if [ $stage -eq 4 ];then
	python3 tools/export.py \
		--config config/voxceleb.yaml \
		--onnx_save_path backone.onnx

	python3 -m onnxsim backone.onnx backone-sim.onnx
fi
