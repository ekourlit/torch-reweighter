import pdb
import argparse
from matplotlib import pyplot as plt
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
NUM_WORKERS = 8
EPOCHS = 3

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
                                         [train_instances, val_instances, test_instances],
                                         generator=torch.Generator().manual_seed(42))

# get dataloaders
train_loader    = DataLoader(ds_train,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS)

val_loader      = DataLoader(ds_val,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS)

test_loader     = DataLoader(ds_test,
                             batch_size=BATCH_SIZE, # can I use here the whole ds? i.e. instances
                             shuffle=False,
                             num_workers=NUM_WORKERS)

#################################################

# init model
model = Conv3DModel(use_batchnorm=False,
                    use_dropout=True)

# log
logger = TensorBoardLogger('logs/', 'test')

# init a trainer
trainer = pl.Trainer(gpus=1,
                     max_epochs=EPOCHS,
                     log_every_n_steps=5,
                     logger=logger)
# train
trainer.fit(model, train_loader, val_loader)