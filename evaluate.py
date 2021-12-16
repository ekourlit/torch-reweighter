import os
import pdb
import argparse
from matplotlib.pyplot import plot
from plotUtils import Plotter, plot_calibration_curve, plot_weights
import random
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
import pytorch_lightning as pl
from models import *
from data import *

#################################################

# set random seeds
def fix_randomness(seed: int, deterministic: bool = False) -> None:
    pl.seed_everything(seed, workers=True)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

fix_randomness(42, True)

def get_flat_array(result_tensor, idx):
    # result_tensor is a list
    # list of length ≃ instances/BATCH_SIZE.
    # each element is a tuple: (probs, weights) 
    # each tuple element is a tensor of shape (BATCH_SIZE, 1)
    # get the elements (probs/weights) as flat np.array in CPU
    elements = map(lambda x: x[idx].cpu().detach().numpy(), result_tensor)
    # first reshape to flatten the arrays to (BATCH_SIZE, ) and then np.concatenate(list)
    elements = map(lambda x: x.reshape(-1), elements)
    elements = np.concatenate(list(elements))
    
    return elements

#################################################
# Arugment parsing

parser = argparse.ArgumentParser(usage="usage: %(prog)s [opts]")
parser.add_argument('-m', '--model', action='store', type=str, dest='model', required=True, help='The model used for evaluation.')
opts = parser.parse_args()
model_path = opts.model

#################################################
# configuration

BATCH_SIZE = 128
NUM_WORKERS = 2

#################################################

# load test dataset
dataset = get_HDF5_dataset('/data/ekourlitis/ILDCaloSim/e-_large/showers-10kE10GeV-RC10-30.hdf5')
dataset_t = get_tensor_dataset(dataset)
# load nominal dataset (just for plotting)
nom_dataset = get_HDF5_dataset('/data/ekourlitis/ILDCaloSim/e-_large/showers-10kE10GeV-RC01-30.hdf5')

# get the labels
labels = np.array(list(map(lambda x: x[1].numpy(), dataset_t))).reshape(-1)

# number of instances/examples
instances = len(dataset_t)
print("Number of instances to predict: %i" % instances)

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
# use GPU1
trainer = pl.Trainer(gpus=[1])
                    #  accelerator='dp')

# inference
result_tensor = trainer.predict(model,
                                test_loader,
                                return_predictions=True)

probs = get_flat_array(result_tensor, 0)
weights = get_flat_array(result_tensor, 1)
# convert any inf to 0
# because score=1 → weight=inf
# score=1 means alternative so I can't re-weight 
# I have lost info on how much it looks like to nominal
weights[np.isinf(weights) == True] = 0.0
# how many zeros?
non_zero_counter = np.count_nonzero(weights==0)
print("Zero weights fraction %0.3f%% " % ( (non_zero_counter/len(weights))*100 ))

# plotting
plots = Plotter(nom_dataset, dataset, weights)
plots.plot_event_edep()
# plots.plot_event_sparcity()
# plot_calibration_curve(labels, probs)
plot_weights(weights)