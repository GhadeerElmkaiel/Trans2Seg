# Folders and Files
## configs
folder contains different configurations for different models (training parameters, dataset parameters, etc.)
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
- #### config
  the code for the configurations used in the code **_cfg_**
- #### data
  dataloder code (here added the sber dataset loaders for different classes)
- #### models
  contains the code for Trans2Seg and Translab neural networks.
- #### modules
  codes used in the neural network
- #### solver
  folder contins the code for solver parameters (lr_scheduler, optimizer, and loss function)
- #### utils
  Not used
## tools
folder contains the codes for training, testing, and inference
## utils
folder contains code for calculating metrics
## workdirs
folders contains the trained models