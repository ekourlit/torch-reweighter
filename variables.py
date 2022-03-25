from numba import njit
import torch
# in general torch layers shape would be: B x C x H x W x D

def calculate_event_energy(layers):
    '''
    Event energy deposit
    '''
    return torch.sum( layers, dim=tuple(d for d in range(1,len(layers.size()))) )

def calculate_non_zero(layers):
    '''
    Event sparcity, i.e non-zero portion
    '''

    sparcities = torch.zeros(layers.size()[0])
    for batch_idx, ilayer in enumerate(layers):
        sparcities[batch_idx] = torch.nonzero(ilayer).size()[0] / (30*30*30)
        
    return sparcities

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