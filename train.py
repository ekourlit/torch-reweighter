import pdb
import argparse
import torch, matplotlib
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from models import Conv3DModel, Conv3DModelGF, MetricsCallback
from data import *
import matplotlib.pyplot as plt
from plotUtils import plot_training_metrics
plt.style.use('default')
font = {'size':14}
matplotlib.rc('font', **font)
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
parser.add_argument('-s', '--save', action='store_true', dest='save', default=False, help='Save the trained model.')
parser.add_argument('-t', '--transform', choices=['None', 'NormPerImg', 'NormGlob', 'LogScale'], default='None', help='Type of transform to perform on the input data (e.g., normalizaing everything to be in the range [0,1]).')
parser.add_argument('-g', '--globalFeatures', nargs='+', default=None, help='The global features to consider as additional inputs. e.g. -g edep sparsity')
parser.add_argument('-n', '--batchNorm', action='store_true', default=False, help='Do batch normalization.')
parser.add_argument('-m', '--modelName', type=str, default='conv3d', help='Name of model.')
parser.add_argument('--stride', type=int, default=3, help='Stride of filter.')
parser.add_argument('-b', '--batchSize',  type=int, default=256, help='Batch size.') #128 is better for atlasgpu
parser.add_argument('-e', '--epochs',  type=int, default=100, help='Max number of epochs to train for.') 
parser.add_argument('-r', '--trainRatio',  type=float, default=0.9, help='Ratio between training and testing set. Default is 0.9.') 
parser.add_argument('-d', '--dataPath', type=str, default='/lcrc/group/ATLAS/atlasfs/local/ekourlitis/ILDCaloSim/e-_Jun3/', help='Path to data files.')
parser.add_argument('-a', '--alt_key', type=str, default='RC10', help='Key for alternative range cut, e.g., RC10.')

opts = parser.parse_args()
save_model = opts.save

#################################################
# configuration
EPOCHS = opts.epochs
# load data into custom Dataset
dataPath = opts.dataPath
#dataPath = '/lcrc/group/ATLAS/atlasfs/local/ekourlitis/ILDCaloSim/e-_large/'
#dataPath = '/lcrc/group/ATLAS/atlasfs/local/ekourlitis/ILDCaloSim/e-_Jun3/'
#dataPath = '/data/ekourlitis/ILDCaloSim/e-_large/partial/'

batchNormStr = ''
if opts.batchNorm:
    batchNormStr = '_batchNorm'

transformStr = ''
transform = None
if opts.transform != 'None':
    transformStr = '_trans'+opts.transform

    if opts.transform == 'NormGlob':
        global_max = get_global_max(dataPath, opts.alt_key)
        transform = locals()[opts.transform](global_max)
    else:
        transform = locals()[opts.transform]()

globalFeaturesStr = ''
if opts.globalFeatures:
    globalFeaturesStr = '_G'+'_G'.join(opts.globalFeatures)

MODELNAME = opts.modelName+f'_stride{opts.stride}_epochs{EPOCHS}'+batchNormStr+transformStr+globalFeaturesStr
BATCH_SIZE = opts.batchSize

NUM_WORKERS = 8
if save_model:
    SAVEPATH = 'models/'
    print("Trained model will be saved at", SAVEPATH)
#################################################

dataset_t = CellsDataset(dataPath, 
                         BATCH_SIZE,
                         transform = transform, # takes None, NormPerImg, NormGlob(scale) or LogScale
                         global_features = opts.globalFeatures, # takes None, edep and/or sparsity
                         alt_key=opts.alt_key)

# number of instances/examples
instances = len(dataset_t)

# split train/val
# the rest will be validation
train_ratio = opts.trainRatio
train_instances = int(train_ratio*instances)
val_instances = int((1-train_ratio)*instances)

# check if the splitting has been done correctly
if instances != train_instances+val_instances:
    delta = instances - (train_instances+val_instances)
    train_instances += delta

print("Train instances: %i" % (train_instances*BATCH_SIZE))
print("Validation instances: %i" % (val_instances*BATCH_SIZE))

ds_train, ds_val = random_split(dataset_t,
                                [train_instances, val_instances],)
                                # generator=torch.Generator().manual_seed(random.randint(0,1e6))) # let's not always train on the same data

# get dataloaders
train_loader    = DataLoader(ds_train,
                             batch_size=None,
                             shuffle=True,
                             num_workers=NUM_WORKERS)

val_loader      = DataLoader(ds_val,
                             batch_size=None,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

#################################################

# get some random training layers
# dataiter = iter(train_loader)
# layers, labels = dataiter.next()
# layers, features, labels = dataiter.next()
# pdb.set_trace()

#################################################

inputShape = next(iter(train_loader))[0].numpy().shape[1:]
print("Shape of input image:", inputShape)
if dataset_t.global_features is not None:
    num_features = next(iter(train_loader))[1].numpy().shape[1:][0]
    print("Number of high-level (global) features:", num_features)

# init model
if dataset_t.global_features is None:
    model = Conv3DModel(inputShape,
                        learning_rate=5e-4,
                        use_batchnorm=opts.batchNorm,
                        use_dropout=True,
                        stride=opts.stride,
                        hidden_layers_in_out=[(512,512),(512,512)])
else:
    model = Conv3DModelGF(inputShape,
                          num_features,
                          learning_rate=5e-4,
                          use_batchnorm=opts.batchNorm,
                          use_dropout=True,
                          stride=opts.stride,
                          hidden_layers_in_out=[(512,512),(512,512)])

# log
logger = TensorBoardLogger('logs/', MODELNAME)

# init a trainer
trainer = pl.Trainer(
                    #  accelerator='cpu',
                     gpus=[0],
                    #  accelerator='ddp',
                     max_epochs=EPOCHS,
                    #  log_every_n_steps=1000,
                     callbacks=[MetricsCallback()],
                     logger=[logger],
                    #  progress_bar_refresh_rate=0,
)
# train
trainer.fit(model, train_loader, val_loader)

# save model
if save_model:
    torch.save(model.state_dict(), SAVEPATH+MODELNAME+'.pt')

# plot some training metrics
# plot_training_metrics(trainer)
