import numpy as np
import h5py
from torch.utils.data import TensorDataset
from torch import Tensor

#################################################

def get_HDF5_dataset(filename: str) -> h5py._hl.group.Group:
    # open the file
    storage_path='/atlasfs02/a/users/ekourlitis/ILDCaloSim/'
    file = h5py.File(storage_path+filename, 'r')
    # get the dataset
    dataset = file[list(file.keys())[0]]

    return dataset

def get_tensor_dataset(dataset: h5py._hl.group.Group, nominal: bool = False) -> TensorDataset:
    # retrieve np arrays
    layers = dataset['layers'][:]
    # reshape layers to add a dummy channel dimension
    layers = layers.reshape(-1, 1, 30, 30, 30)
    energy = dataset['energy'][:]
    if not nominal:
        labels = np.ones(energy.shape)
    else:
        labels = np.zeros(energy.shape)

    # convert to tensors
    layers_t = Tensor(layers)
    labels_t = Tensor(labels)

    # create torch dataset
    dataset_t = TensorDataset(layers_t, labels_t)

    return dataset_t