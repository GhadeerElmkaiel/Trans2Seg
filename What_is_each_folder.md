# Folders and Files
## configs
folder contains different configurations for different models (training parameters, dataset parameters, etc.)
```trans2seg_medium_all_sber_merged.yaml``` is the configuration file for the SberMerged (2400+3500) dataset
## datasets
folder contains the used datasets
## debug
Not used
## docs
Not used
## runs
Conatins the results of inference
## segmentron
the main code files for the model's structure and the dataset
- #### **config**
  the code for the configurations used in the code **_cfg_**
  ```config.py``` the configuration functions and structure
- #### **data**
  dataloder code (here added the sber dataset loaders for different classes)
  ```sber_merged_dataset_all_classes.py``` is the code of the dataloader for Sber merged dataset (2400+3500)
- #### **models**
  contains the code for Trans2Seg and Translab neural networks.
  - #### **backbones**
    contains the codes for multiple backbones.
  ```segbase.py``` is the base class for **Trans2Seg** and **TransLab**
- #### **modules**
  codes used in the neural network such as *drop* and *norm*
- #### **solver**
  folder contins the code for solver parameters (lr_scheduler, optimizer, and loss function)
  - ```loss.py``` descripes the loss functions
  - ```lovasz_losses.py``` code for lovasz loss function
  - ```lr_scheduler.py``` for learning rate
- #### **utils**
  Not used
## tools
folder contains the codes for training, testing, and inference
```train.py``` code for training
```demo.py``` code for inference
## utils
folder contains code for calculating metrics
## workdirs
folders contains the trained models