#!/bin/bash

voxceleb1_path=/ssd/VoxCeleb/voxceleb1
voxceleb2_path=/ssd/VoxCeleb/voxceleb2

stage=2

if [ $stage -eq 0 ];then
    rm -rf data/train/
    mkdir -p data/train/

    ln -s ${voxceleb1_path}/vox1_dev_wav/* data/train/
    ln -s ${voxceleb2_path}/dev/aac/* data/train

    wget https://openslr.magicdatatech.com/resources/49/voxceleb1_test_v2.txt
    mv voxceleb1_test_v2.txt data/voxceleb1_test_v2.txt
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
	--src_trials_path data/voxceleb1_test_v2.txt \
	--dst_trials_path data/trial.lst
fi

if [ $stage -eq 2 ];then
    python3 tools/train.py \
        --config config/vox_baseline.yaml
fi

if [ $stage -eq 3 ];then
    python3 tools/evaluate.py \
       --config config/vox_baseline.yaml
fi

