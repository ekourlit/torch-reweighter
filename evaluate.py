import pdb
import argparse
import random
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models import *
from data import *

#################################################

# set random seeds
def fix_randomness(seed: int, deterministic: bool = False) -> None:
    pl.seed_everything(seed, workers=True)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

fix_randomness(42, False)

#################################################
# Arugment parsing

parser = argparse.ArgumentParser(usage="usage: %(prog)s [opts]")
parser.add_argument('-m', '--model', action='store', type=str, dest='model', required=True, help='The model used for evaluation.')
opts = parser.parse_args()
model_path = opts.model

#################################################
# configuration

BATCH_SIZE = 1
NUM_WORKERS = 16

#################################################

# load datasets
dataset = get_HDF5_dataset('showers-10kPhot1GeV_calo_10.hdf5')
dataset_t = get_tensor_dataset(dataset)

# number of instances/examples
instances = len(dataset_t)

test_loader = DataLoader(dataset_t,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

#################################################
'''
# get some random training images
dataiter = iter(test_loader)
images, labels = dataiter.next()
pdb.set_trace()
'''
#################################################

# init model
model = Conv3DModel()
model.load_state_dict(torch.load(model_path))

# init a trainer
trainer = pl.Trainer(gpus=1)

# inference
result_tensor = trainer.predict(model,
                                test_loader,
                                return_predictions=True)

# get the results/weights as flat np.array in CPU
weights = map(lambda x: x.cpu().detach().numpy(), result_tensor)
weights = np.array(list(results)).reshape(-1)

# todo
# write some routines to plot un-weighted/weighted
# think FROM where TO where the weight drives you