import pdb
import argparse
from operator import itemgetter
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from models import Conv3DModel
from data import CellsDataset, Scale

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
parser.add_argument('-n', '--batchNorm', action='store_true', default=False, help='Do batch normalization.')
parser.add_argument('-m', '--modelName', type=str, default='conv3d', help='Name of model.')
parser.add_argument('--stride', type=int, default=1, help='Stride of filter.')
parser.add_argument('-b', '--batchSize',  type=int, default=256, help='Batch size.') #128 is better for atlasgpu
opts = parser.parse_args()
save_model = opts.save

#################################################
# configuration

batchNormStr = ''
if opts.batchNorm:
    batchNormStr = '_batchNorm'
MODELNAME = opts.modelName+f'_stride{opts.stride}'+batchNormStr
BATCH_SIZE = opts.batchSize

NUM_WORKERS = 8
EPOCHS = 100
if save_model:
    SAVEPATH = 'models/'
    print("Trained model will be saved at", SAVEPATH)
#################################################

# load data into custom Dataset
dataset_t = CellsDataset('/data/ekourlitis/ILDCaloSim/e-_large/all/', 
                         BATCH_SIZE,
                         transform = Scale())

# number of instances/examples
instances = len(dataset_t)

# split train/val
# the rest will be validation
train_ratio = 0.99
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
# pdb.set_trace()

#################################################

inputShape = next(iter(train_loader))[0].numpy().shape[1:]
print("Shape of input:", inputShape)

# init model
model = Conv3DModel(inputShape,
                    learning_rate=5e-4,
                    use_batchnorm=opts.batchNorm,
                    use_dropout=True,
                    stride=opts.stride)

# log
logger = TensorBoardLogger('logs/', MODELNAME)
csvLogger = CSVLogger("logs/", name=MODELNAME+'_csv')

# init a trainer
trainer = pl.Trainer(#accelerator='cpu',
                     gpus=[0],
                    #  accelerator='ddp',
                     max_epochs=EPOCHS,
                     log_every_n_steps=1000,
                     logger=[logger, csvLogger],
    #progress_bar_refresh_rate=0,
)
# train
trainer.fit(model, train_loader, val_loader)

# save model
if save_model:
    torch.save(model.state_dict(), SAVEPATH+MODELNAME+'.pt')
