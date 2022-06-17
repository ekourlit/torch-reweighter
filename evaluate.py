import os, re
import pdb
import argparse
from matplotlib.pyplot import plot
from plotUtils import *#Plotter, plot_calibration_curve, plot_weights
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
parser.add_argument('-n', '--batchNorm', action='store_true', default=False, help='Do batch normalization.')
parser.add_argument('--stride', type=int, default=3, help='Stride of filter')
parser.add_argument('-b', '--batchSize',  type=int, default=256, help='Batch size') #128 is better for atlasgpu
parser.add_argument('-l', '--logVersion',  type=str, default='version_0', help='Version of the log to use for plotting') 

opts = parser.parse_args()
model_path = opts.model

#################################################
# configuration

BATCH_SIZE = opts.batchSize
NUM_WORKERS = 2
MODELNAME=model_path.split('/')[-1].rstrip('pt').rstrip('.')
batchNormStr = ''
if opts.batchNorm:
    batchNormStr = '_batchNorm'
suffix = f'_stride{opts.stride}'+batchNormStr
GLOBAL_FEATURES = []
for i in MODELNAME.split('_'):
    if 'G' in i: GLOBAL_FEATURES.append(i.lstrip('G'))

#################################################

# load test dataset
dataset = get_HDF5_dataset('/data/whopkins/e-_Jun3/test/showers-10kE10GeV-RC10-95.hdf5')
dataset_t = get_tensor_dataset(dataset, GLOBAL_FEATURES)

#dataset = get_HDF5_dataset('/data/ekourlitis/ILDCaloSim/e-_large/test/showers-10kE10GeV-RC10-95.hdf5')
#dataset_t = get_tensor_dataset(dataset, GLOBAL_FEATURES)
# load nominal dataset (just for plotting)
nom_dataset = get_HDF5_dataset('/data/whopkins/e-_Jun3/showers-10kE10GeV-RC01-30.hdf5')

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
# images, labels = dataiter.next()
images, features, labels = dataiter.next()
pdb.set_trace()
'''
#################################################

# init model
inputShape = next(iter(test_loader))[0].numpy().shape[1:]
print("Shape of input:",inputShape)
if len(GLOBAL_FEATURES):
    num_features = next(iter(test_loader))[1].numpy().shape[1:][0]
    print("Number of high-level (global) features:", num_features)

# init model
if len(GLOBAL_FEATURES):
    model = Conv3DModelGF(inputShape,
                          num_features,
                          use_batchnorm=opts.batchNorm,
                          use_dropout=True,
                          stride=opts.stride,
                          hidden_layers_in_out=[(512,512), (512,512)]
                          )

else:
    model = Conv3DModel(inputShape,
                        use_batchnorm=opts.batchNorm,
                        use_dropout=True,
                        stride=opts.stride,
                        hidden_layers_in_out=[(512,512), (512,512)]
                        )

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

# clip on maxWeight
#maxWeight = 500
#weights[weights > maxWeight] = 0

# how many zeros?
zero_counter = np.count_nonzero(weights==0)
print("Zero weights fraction: %0.3f%% " % ( (zero_counter/len(weights))*100 ))

#################################################
# Plotting

plots = Plotter(nom_dataset, dataset, weights)
# plot_weights(weights, suffix=suffix)
plots.plot_event_observables(suffix=suffix)

# plot_calibration_curve(labels, probs)

# csvLoggerPath = "logs/"+MODELNAME+'_csv/'+opts.logVersion+'/metrics.csv'
# print(csvLoggerPath)
# plot_metrics(csvLoggerPath, suffix=suffix)
