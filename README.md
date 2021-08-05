# Transparent Transformer Segmentation
## Introduction
This repository contains the data and code for "Trans2Seg: Transparent Object Segmentation with Transformer 	".


## Environments

- python 3
- torch = 1.4.0
- torchvision
- pyyaml
- Pillow
- numpy

## INSTALL

```
python setup.py develop --user
```
## Additionl for Sber
After the installation a ```run_num.log``` should be created with a number inside it (for example 87), this number is used to name the log file on **wandb**  
also wandb should be installed:  
```
pip install wandb
``` 

## Data Preparation
1. create dirs './datasets/transparent/Trans10K_v2' 
2. put the train/validation/test data under './datasets/transparent/Trans10K_v2'. 
Data Structure is shown below.
```
Trans10K_v2
├── test
│   ├── images
│   └── masks_12
├── train
│   ├── images
│   └── masks_12
└── validation
    ├── images
    └── masks_12
```
Download Dataset: [Baidu Drive](https://pan.baidu.com/s/1P-2l-Q2brbnwRd2kXi--Dg). code: oqms

## Instructions for Sberbank robotics lab:
Please se the [how_to_use](how_to_use.md) file for detailed instructions about training on new dataset and applying inference. 

A discription for the usage of each folder can be found in this [file](What_is_each_folder.md)
## Evaluation

## Network Define
The code of Network pipeline is in `segmentron/models/trans2seg.py`.

The code of Transformer Encoder-Decoder is in `segmentron/modules/transformer.py`.

## Train
Our experiments are based on one machine with 8 V100 GPUs with 32g memory, about 1 hour training time.

```
bash tools/dist_train.sh $CONFIG-FILE $GPUS
```

For example:
```
bash tools/dist_train.sh configs/trans10kv2/trans2seg/trans2seg_medium.yaml
```

## Test
```
bash tools/dist_train.sh $CONFIG-FILE $GPUS --test TEST.TEST_MODEL_PATH $MODEL_PATH
```


## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```
@article{xie2021segmenting,
  title={Segmenting transparent object in the wild with transformer},
  author={Xie, Enze and Wang, Wenjia and Wang, Wenhai and Sun, Peize and Xu, Hang and Liang, Ding and Luo, Ping},
  journal={arXiv preprint arXiv:2101.08461},
  year={2021}
}
```
