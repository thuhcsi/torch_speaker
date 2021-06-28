#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
voxceleb1_path=~/datasets/VoxCeleb/voxceleb1
checkpoint_path=ckpt.pt

stage=2

if [ $stage -eq 1 ];then
	rm -rf data; mkdir data
	wget -P data/ https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
	echo format trails
	python3 scripts/format_trials.py \
		--voxceleb1_root $voxceleb1_path \
		--src_trials_path data/veri_test.txt \
		--dst_trials_path data/vox1.txt
fi

if [ $stage -eq 2 ];then
	python3 tools/evaluate.py \
		--config config/voting.yaml \
		--trial_path data/vox1.txt \
		--checkpoint_path $checkpoint_path
fi

if [ $stage -eq 3 ];then
	python3 local/attack.py \
		--config config/voting.yaml \
		--trial_path data/vox1.txt \
		--checkpoint_path $checkpoint_path
fi

if [ $stage -eq 3 ];then
	python3 local/defense.py \
		--config config/voting.yaml \
		--trial_path data/vox1.txt \
		--checkpoint_path $checkpoint_path
fi

