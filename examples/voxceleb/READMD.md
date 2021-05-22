## Data Preparation

```
ln -s your_voxceleb1_path/vox1_dev_wav/* data/train/
ln -s your_voxceleb2_path/dev/aac/* data/train

wget https://openslr.magicdatatech.com/resources/49/voxceleb1_test_v2.txt
mv voxceleb1_test_v2.txt data/voxceleb1_test_v2.txt

python3 scripts/build_datalist.py \
    --extension wav \
    --dataset_dir data/train \
    --data_list_path data/train.csv

python3 scripts/format_trials.py \
    --voxceleb1_root $voxceleb1_path \
    --src_trials_path data/voxceleb1_test_v2.txt \
    --dst_trials_path data/trial.lst
```

## Training

```
python3 tools/train.py \
    --config your_yaml_path
```

## Evaluation

```
python3 evaluate.py \
    --config your_yaml_path
```
