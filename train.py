import pdb
import argparse
from matplotlib import pyplot as plt
import random
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models import *
from data import *

#################################################
# configuration

BATCH_SIZE = 32
NUM_WORKERS = 16
EPOCHS = 200

#################################################

# set random seeds
def fix_randomness(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

fix_randomness(42)

#################################################

# load datasets
dataset_nom = get_HDF5_dataset('showers-10kPhot1GeV_calo_01.hdf5')
dataset_t_nom = get_tensor_dataset(dataset_nom, nominal=True)

dataset_alt = get_HDF5_dataset('showers-10kPhot1GeV_calo_10.hdf5')
dataset_t_alt = get_tensor_dataset(dataset_alt)

# concatenate tensor datasets
dataset_t = ConcatDataset([dataset_t_nom, dataset_t_alt])

# number of instances/examples
instances = len(dataset_t)

# split train/val/test
# the rest will be validation
train_ratio = 0.7
test_ratio = 0.01
train_instances = int(train_ratio*instances)
val_instances = int((1-train_ratio-test_ratio)*instances)
test_instances = int(test_ratio*instances)

# check if the splitting has been done correctly
if instances != train_instances+val_instances+test_instances:
    delta = instances - (train_instances+val_instances+test_instances)
    train_instances += delta

ds_train, ds_val, ds_test = random_split(dataset_t,
                                         [train_instances, val_instances, test_instances])

# get dataloaders
train_loader    = DataLoader(ds_train,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS)

val_loader      = DataLoader(ds_val,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

test_loader     = DataLoader(ds_test,
                             batch_size=BATCH_SIZE, # can I use here the whole ds? i.e. instances
                             shuffle=False,
                             num_workers=NUM_WORKERS)

#################################################
'''
# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
pdb.set_trace()
'''
#################################################

# init model
model = Conv3DModel(learning_rate=1e-4,
                    use_batchnorm=False,
                    use_dropout=True)

# log
logger = TensorBoardLogger('logs/', 'conv3d')

# init a trainer
trainer = pl.Trainer(gpus=1,
                     max_epochs=EPOCHS,
                     log_every_n_steps=5,
                     logger=logger)
# train
trainer.fit(model, train_loader, val_loader)