import pdb
import argparse
import torch
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models import Conv3DModel
from data import CellsDataset

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
opts = parser.parse_args()
save_model = opts.save

#################################################
# configuration

MODELNAME = 'conv3d_large1_1GPU_test'
BATCH_SIZE = 128
NUM_WORKERS = 16
EPOCHS = 2
if save_model:
    SAVEPATH = 'models/'
    print("Trained model will be saved at", SAVEPATH)

#################################################

# load data into custom Dataset
dataset_t = CellsDataset('/atlasfs02/a/users/ekourlitis/ILDCaloSim/e-/e-_large/', BATCH_SIZE)

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
'''
# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
pdb.set_trace()
'''
#################################################

# init model
model = Conv3DModel(learning_rate=1e-3,
                    use_batchnorm=True,
                    use_dropout=True)

# log
logger = TensorBoardLogger('logs/', MODELNAME)

# init a trainer
trainer = pl.Trainer(gpus=[1],
                    #  accelerator='ddp',
                     max_epochs=EPOCHS,
                     log_every_n_steps=5,
                     logger=logger)
# train
trainer.fit(model, train_loader, val_loader)

# save model
if save_model:
    torch.save(model.state_dict(), SAVEPATH+MODELNAME+'.pt')