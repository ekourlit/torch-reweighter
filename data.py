import os
import pdb
import numpy as np
import h5py
from torch.utils.data import TensorDataset, Dataset, ConcatDataset, random_split
from torch import FloatTensor
import torchvision.transforms as transforms
from typing import Tuple
from math import floor

#################################################

def get_HDF5_dataset(filename: str) -> h5py._hl.group.Group:
    # open the file
    # storage_path='/atlasfs02/a/users/ekourlitis/ILDCaloSim/'
    file = h5py.File(filename, 'r')
    # get the dataset
    dataset = file[list(file.keys())[0]]

    return dataset

def get_tensor_dataset(dataset: h5py._hl.group.Group, nominal: bool = False) -> TensorDataset:
    # retrieve np arrays
    layers = dataset['layers'][:]
    energy = dataset['energy'][:]
    # reshape layers to add a dummy channel dimension
    layers = layers.reshape(-1, 1, 30, 30, 30)
    if not nominal:
        labels = np.ones(energy.shape)
    else:
        labels = np.zeros(energy.shape)

    # convert to tensors
    layers_t = FloatTensor(layers)
    labels_t = FloatTensor(labels)

    # create torch dataset
    dataset_t = TensorDataset(layers_t, labels_t)

    # normalize to [-1, 1] -> that's wrong...
    dataset_t.transform = transforms.Compose(
        [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return dataset_t

class CellsDataset(Dataset):
    # TODO: need to add [0,1] normalization
    # currently using BatchNorm in the first layer of the model -> slow training

    def __init__(self, 
                 path: str,
                 batch_size: int,
                 transform: bool = None,
                 nom_key: str = 'RC01', 
                 alt_key: str = 'RC10') -> None:

        self.storage_path = path
        self.files = os.listdir(self.storage_path)
        self.files.sort(key=self.count_sorter)
        self.batch_size = batch_size
        self.transform = transform
        self.nom_key = nom_key
        self.alt_key = alt_key
        self.events_per_file = []
        self.cumulative_events_per_file = []
    
    def count_sorter(self, name: str) -> str:
        # e.g. showers-10kE10GeV-RC10-123.hdf5
        num_tag = int(name.split('.')[0].split('-')[-1])

        return "{0:0=3d}".format(num_tag)

    def event_counter(self, filename: str) -> int:
        try:
            with h5py.File(self.storage_path+filename, 'r') as file:
                dataset = file[list(file.keys())[0]]
                events = len(dataset['energy'])
        # should be able to catch now if there is any I/O problem with any file
        except BaseException as err:
            print("Unexpected Error: %s of type %s" % (err, type(err)))
            raise

        return events

    def __len__(self) -> int:
        self.events_per_file = list(map(self.event_counter, self.files))
        total_events = sum(self.events_per_file)
        # construct cumulative event list
        events = 0
        for i in self.events_per_file:
            events += i
            self.cumulative_events_per_file.append(events)

        return floor(total_events/self.batch_size)

    # NB: randomization happens at the idx
    def __getitem__(self, idx) -> Tuple[FloatTensor, FloatTensor]:
        # initial event idx
        init_evnt_idx = idx*self.batch_size
        # find into which file the event belongs
        file_idx = np.nonzero(np.array(self.cumulative_events_per_file) > init_evnt_idx)[0][0]
        filename = self.files[file_idx]
        # RC-tag e.g. showers-10kE10GeV-RC10-123.hdf5 -> RC10
        RCtag = filename.split('.')[0].split('-')[-2]
        # is nominal or alternative?
        if RCtag == self.nom_key:
            isNominal = True
        else:
            isNominal = False
        
        instance_idx = init_evnt_idx if file_idx == 0 else init_evnt_idx - self.cumulative_events_per_file[file_idx-1]
        # get data lazily
        with h5py.File(self.storage_path+filename, 'r') as file:
            dataset = file[list(file.keys())[0]]
            # retrieve np arrays
            layers = dataset['layers'][instance_idx:instance_idx+self.batch_size]
            energy = dataset['energy'][instance_idx:instance_idx+self.batch_size]
            # reshape layers to add a dummy channel dimension
            layers = layers.reshape(-1, 1, 30, 30, 30)
            if not isNominal:
                label = np.ones(energy.shape)
            else:
                label = np.zeros(energy.shape)
            # convert to tensors
            layers = FloatTensor(layers)
            label = FloatTensor(label)
            # apply any extra transform
            if self.transform:
                layers = self.transform(layers)

            return layers, label