# Train on our dataset.
To train on our dataset **Sber2400 Dataset** it is necessary to create a dataset loader for this dataset in 
```bash
./data/dataloader/sber_dataset.py               #Optical Objects and Floor
./data/dataloader/sber_dataset_all_classes.py   #All Classes
```
## Add new DataLoader
in **sber_dataset.py** I copied **transparent11.py** and I did the following changes:

- Changed the root to the dataset
- in *_get_sber2400_pairs* I Changed the pathes to the masks. Also changed the extentions for the images.
- I used adaptive color palette while converting the mask image.
```python
# Note this change did not work, because for 
# each image this will create a new palette according
# to the colors in each image
mask = Image.open(self.masks[index]).convert("P", palette=Image.ADAPTIVE)
# mask = Image.open(self.masks[index]).convert("P")

```

So when I initilize the dataset class I add this code to read a palette image and define a fixed palette for the dataset

```python
############ Important ##############
# Change the palette for each dataset
self.src_palette = Image.open(root+"all_palette.png")
self.src_palette = self.src_palette.convert("P", palette=Image.ADAPTIVE)
############ ######### ##############
```

Then instead of doing image convert, I used quantize as follows:

```python
# mask = Image.open(self.masks[index]).convert("P", palette=Image.ADAPTIVE)
mask = Image.open(self.masks[index]).quantize(palette=self.src_palette)
```

The adaptive palette is so I don't need to manumally set a converter for each collor.

## Add new config file
I created a config file and changed the name of the used dataset to be "sber_dataset" which is consistent with the used dataloader. *note: this can be done by adding anouther argument and use it to pass the name of the dataset*

## Trian the NN:
To train the NN I use one of the following codes 

```bash
bash tools/dist_train.sh configs/trans10kv2/trans2seg/trans2seg_medium.yaml # original Trans10KV2 dataset
bash tools/dist_train.sh configs/trans10kv2/trans2seg/trans2seg_medium_sber.yaml # All optical and Floor
bash tools/dist_train.sh configs/trans10kv2/trans2seg/trans2seg_medium_all_sber.yaml # All classes In Sber dataset
```
The dataset and training parameters can be edited in the config files

## Inference:
To do inference on different images run the following code:
```bash
CUDA_VISIBLE_DEVICES=0 python -u ./tools/demo.py --config-file [path to the config file] --input-img [path to the test dataset] TEST.TEST_MODEL_PATH [path to the trained model]
```
for example:
```bash
CUDA_VISIBLE_DEVICES=0 python -u ./tools/demo.py --config-file configs/trans10kv2/trans2seg/trans2seg_medium.yaml --input-img ./datasets/Sber2400/test/images/ TEST.TEST_MODEL_PATH ./workdirs/trans10kv2/trans2seg_medium/50.pth

CUDA_VISIBLE_DEVICES=0 python -u ./tools/demo.py --config-file configs/trans10kv2/trans2seg/trans2seg_medium.yaml --input-img ./datasets/Sber3500/images/test/images/ TEST.TEST_MODEL_PATH ./workdirs/trans10kv2/trans2seg_medium/Sber2400_50_All_classes.pth DATASET.NAME sber_dataset_all
```
