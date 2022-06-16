from numba import njit
import torch
import numpy as np
from numba import njit, prange
import math
from typing import Tuple

'''
-----------------------------------------------------
Calculations for plotting - operating on numpy arrays
-----------------------------------------------------
'''

@njit(parallel=True)
def calculate_edep_np(max_events: int, layers: np.ndarray) -> np.ndarray:
    events_energy_deposit = np.zeros(max_events)

    for event in prange(max_events):
        event_layers = layers[event, :, :, :]
        events_energy_deposit[event] = np.sum(event_layers)
    
    return events_energy_deposit

@njit(parallel=True)
def calculate_non_zero_np(max_events: int, layers: np.ndarray) -> np.ndarray:
    nonzero_portions = np.zeros(max_events)
    total_cells = 30*30*30

    for event in prange(max_events):
        event_layers = layers[event, :, :, :]
        nonzero_elements = len(np.nonzero(event_layers)[0])
        nonzero_portion = nonzero_elements/total_cells
        nonzero_portions[event] = nonzero_portion
    
    return nonzero_portions

@njit(parallel=True)
def calculate_longitudinal_centroid_np(max_events: int, layers: np.ndarray) -> np.ndarray:
    event_lcentroid = np.zeros(max_events)
    
    # loop over events
    for event in prange(max_events):
        event_layers = layers[event, :, :, :]
        energies_per_layer = np.zeros(30)
        ewgt_idx_per_layer = np.zeros(30)
        
        # loop over y-layers
        for i in prange(30):
            ylayer = event_layers[i, :, :]
            layer_energy = np.sum(ylayer)
            layer_idx = i+1
            ewgt_idx = layer_idx*layer_energy
            
            ewgt_idx_per_layer[i] = ewgt_idx
            energies_per_layer[i] = layer_energy
        
        event_energy = np.sum(energies_per_layer)
        event_lcentroid[event] = np.sum(ewgt_idx_per_layer / event_energy)
    
    return event_lcentroid

@njit(parallel=True)
def calculate_com(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w_y = w_x = w_z = np.zeros(30*30*30)
    norm = 0
    
    # loop over cells
    i = 0
    for y in prange(30):
        for x in prange(30):
            for z in prange(30):
                edep = array[y, x, z]
                w_y[i]= edep*y
                w_x[i]= edep*x
                w_z[i]= edep*z
                norm += edep
                i += 1
    
    w_y = w_y/norm
    w_x = w_x/norm
    w_z = w_z/norm
    
    y_com = np.sum(w_y)
    x_com = np.sum(w_x)
    z_com = np.sum(w_z)
    
    return y_com, x_com, z_com

@njit(parallel=True)
def calculate_r2_np(max_events: int, layers: np.ndarray) -> np.ndarray:
    event_r2 = np.zeros(max_events)
    
    # loop over events
    for event in prange(max_events):
        event_layers = layers[event, :, :, :]
        # caclulate center of mass
        y_com, x_com, z_com = calculate_com(event_layers)
        
        event_edep = 0
        event_ewgt_r2distance = 0

        # loop over cells
        for y in prange(30):
            for x in prange(30):
                for z in prange(30):
                    edep = event_layers[y,x,z]
                    y_cell = y
                    x_cell = x
                    z_cell = z
                    r_distance = math.sqrt((x_cell**2 + z_cell**2)) - math.sqrt((x_com**2 + z_com**2))
                    ewgt_r2distance = edep*(r_distance**2)

                    event_edep += edep
                    event_ewgt_r2distance += ewgt_r2distance

        event_r2[event] = (event_ewgt_r2distance / event_edep)
    
    return event_r2

@njit(parallel=True)
def calculate_Rz_np(max_events: int, layers: np.ndarray) -> np.ndarray:
    event_Rz = np.zeros(max_events)

    # loop over events
    for event in prange(max_events):
        edep = np.sum(layers[event, :, :, :])
        Rz = np.zeros(30)

        # loop over layers
        for i in prange(30):
            layer = layers[event, i, :, :]
            layer_edep = np.sum(layer)
            num = np.sum(layer[18:27, 13:17])
            denom = np.sum(layer[18:27, 11:19])
            Rz_ewgted = ((num/denom)*layer_edep) if denom else 0
            Rz[i] = Rz_ewgted

        # get the sum of Rz and normalize
        event_Rz[event] = np.sum(Rz)/edep

    return event_Rz

@njit(parallel=True)
def calculate_Rx_np(max_events: int, layers: np.ndarray) -> np.ndarray:
    event_Rx = np.zeros(max_events)
    
    # loop over events
    for event in prange(max_events):
        edep = np.sum(layers[event, :, :, :])
        Rx = np.zeros(30)

        # loop over layers
        for i in range(30):
            layer = layers[event, i, :, :]
            layer_edep = np.sum(layer)
            num = np.sum(layer[20:25, 12:18])
            denom = np.sum(layer[17:28, 12:18])
            Rx_ewgted = ((num/denom)*layer_edep) if denom else 0
            Rx[i] = Rx_ewgted

        # get the sum of Rx and normalize
        event_Rx[event] = np.sum(Rx)/edep

    return event_Rx

@njit(parallel=True)
def calculate_lambda2_np(max_events: int, layers: np.ndarray) -> np.ndarray:
    event_lambda2 = np.zeros(max_events)
    
    # loop over events
    for event_num in prange(max_events):
        event_layers = layers[event_num, :, :, :]
        # caclulate center of mass
        y_com, x_com, z_com = calculate_com(event_layers)
        
        event_edep = 0
        event_ewgt_l2distance = 0

        # loop over cells
        for y in prange(30):
            for x in prange(30):
                for z in prange(30):
                    edep = event_layers[y,x,z]
                    y_cell = y
                    x_cell = x
                    z_cell = z
                    l_distance = y_cell - y_com
                    ewgt_l2distance = edep*(l_distance**2)

                    event_edep += edep
                    event_ewgt_l2distance += ewgt_l2distance

        event_lambda2[event_num] = event_ewgt_l2distance / event_edep
    
    return event_lambda2

'''
------------------------------------------------------
Calculations for training - operating on torch tensors
------------------------------------------------------

in general torch layers shape would be: B x C x H x W x D
'''

def calculate_event_energy(layers: torch.FloatTensor) -> torch.FloatTensor:
    '''
    Event energy deposit
    '''
    return torch.sum( layers, dim=tuple(d for d in range(1,len(layers.size()))) )

def calculate_non_zero(layers: torch.FloatTensor) -> torch.FloatTensor:
    '''
    Event sparsity, i.e non-zero portion
    '''

    sparcities = torch.zeros(layers.size()[0])
    for batch_idx, ilayer in enumerate(layers):
        sparcities[batch_idx] = torch.nonzero(ilayer).size()[0] / (30*30*30)
        
    return sparcities

global_features_funcs = {'edep': calculate_event_energy, 
                        'sparsity': calculate_non_zero}

def calculate_longitudinal_centroid(layers):
    '''
    Event longitudinal centrod in the sense of how shower evolves, i.e. y-axix or array rows (axis=0)
    WIP!
    '''

    tot_events = len(layers)
    event_lcentroid = np.zeros(tot_events)
    
    # loop over events
    for event_num in range(tot_events):
        event_layers = layers[event_num, :, :, :]
        energies_per_layer = np.zeros(30)
        ewgt_idx_per_layer = np.zeros(30)
        
        # loop over y-layers
        for i in range(30):
            ylayer = event_layers[i, :, :]
            layer_energy = np.sum(ylayer)
            layer_idx = i+1
            ewgt_idx = layer_idx*layer_energy
            
            ewgt_idx_per_layer[i] = ewgt_idx
            energies_per_layer[i] = layer_energy
        
        event_energy = np.sum(energies_per_layer)
        event_lcentroid[event_num] = np.sum(ewgt_idx_per_layer / event_energy)
    
    return event_lcentroid

@njit(parallel=True)
def calculate_r2(layers):
    '''
    shower shape (r2)
    WIP!
    '''
    tot_events = layers.shape[0]
    event_r2 = np.zeros(tot_events)
    
    # loop over events
    for event_num in prange(tot_events):
        event_layers = layers[event_num, :, :, :]
        # caclulate center of mass
        y_com, x_com, z_com = calculate_com(event_layers)
        
        event_edep = 0
        event_ewgt_r2distance = 0

        # loop over cells
        for y in prange(30):
            for x in prange(30):
                for z in prange(30):
                    edep = event_layers[y,x,z]
                    y_cell = y
                    x_cell = x
                    z_cell = z
                    r_distance = math.sqrt((x_cell**2 + z_cell**2)) - math.sqrt((x_com**2 + z_com**2))
                    ewgt_r2distance = edep*(r_distance**2)

                    event_edep += edep
                    event_ewgt_r2distance += ewgt_r2distance

        event_r2[event_num] = event_ewgt_r2distance / event_edep
    
    return event_r2

def calculate_Rz(layers):
    '''
    Rz - layer cell's ratio
    WIP!
    '''
    event_Rz = []
    # loop over variant events
    for event in range(layers.shape[0]):
        edep = np.sum(layers[event, :, :, :])
        Rz = []
        # loop over layers
        for i in range(30):
            layer = layers[event, i, :, :]
            layer_edep = np.sum(layer)
            num = np.sum(layer[18:27, 13:17])
            denom = np.sum(layer[18:27, 11:19])
            Rz_ewgted = ((num/denom)*layer_edep) if denom else 0
            Rz.append(Rz_ewgted)
        # get the sum of Rz and normalize
        event_Rz.append(np.sum(Rz)/edep)

    return np.array(event_Rz)

def calculate_Rx(layers):
    '''
    Rx - layer cell's ratio
    WIP!
    '''
    event_Rx = []
    # loop over variant events
    for event in range(layers.shape[0]):
        edep = np.sum(layers[event, :, :, :])
        Rx = []
        # loop over layers
        for i in range(30):
            layer = layers[event, i, :, :]
            layer_edep = np.sum(layer)
            num = np.sum(layer[20:25, 12:18])
            denom = np.sum(layer[17:28, 12:18])
            Rx_ewgted = ((num/denom)*layer_edep) if denom else 0
            Rx.append(Rx_ewgted)
        # get the sum of Rx and normalize
        event_Rx.append(np.sum(Rx)/edep)

    return np.array(event_Rx)