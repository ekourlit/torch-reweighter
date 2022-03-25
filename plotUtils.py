from re import A
import numpy as np
from matplotlib import pyplot as plt
import h5py
from datetime import date
today = str(date.today())
from os import system
import pdb
from sklearn.calibration import calibration_curve
from scipy.stats import wasserstein_distance, sem
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
plt.style.use('default')
font = {'size':14}
matplotlib.rc('font', **font)
from esutil.stat import wmom
import pandas as pd
from numba import njit,jit, prange
import math

@njit(parallel=True)
def calculate_com(array):
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
def really_calculate_r2(layers, max_events):
    event_r2 = np.zeros(max_events)
    
    # loop over events
    for event_num in prange(max_events):
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

        event_r2[event_num] = (event_ewgt_r2distance / event_edep)
    
    return event_r2

@njit(parallel=True)
def really_calculate_lambda2(layers, max_events):
    tot_events = max_events
    event_lambda2 = np.zeros(tot_events)
    
    # loop over events
    for event_num in prange(tot_events):
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

class Plotter:
    """
    Class to create plots. 
    The different kind of plots are created by the class methods.
    """

    def __init__(self,
                 nominal_dataset: h5py._hl.group.Group, 
                 dataset: h5py._hl.group.Group, 
                 weights: np.ndarray) -> None:

        self.max_events = 9000
        # construct np.arrays
        self.nominal_layers = nominal_dataset['layers'][:][:self.max_events, :, :, :] # shape: (self.max_events, 30, 30, 30)
        self.layers = dataset['layers'][:][:self.max_events, :, :, :] # shape: (self.max_events, 30, 30, 30)
        self.weights = 1./weights[:self.max_events] # shape: (self.max_events, )

        self.saveDir = 'plots/'+today
        system('mkdir -p '+self.saveDir)

    def calculate_edep(self, layers: np.ndarray) -> list:
        events_energy_deposit = []
        for event in range(self.max_events):
            event_layers = layers[event, :, :, :]
            events_energy_deposit.append(np.sum(event_layers))
        
        return events_energy_deposit

    def calculate_non_zero(self, layers: np.ndarray) -> list:
        nonzero_portions = []
        total_cells = 30*30*30
        for event in range(self.max_events):
            event_layers = layers[event, :, :, :]
            nonzero_elements = len(np.nonzero(event_layers)[0])
            nonzero_portion = nonzero_elements/total_cells
            nonzero_portions.append(nonzero_portion)
        
        return nonzero_portions

    def calculate_longitudinal_centroid(self, layers: np.ndarray) -> list:
        event_lcentroid = []
        
        # loop over events
        for event_num in range(self.max_events):
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
            event_lcentroid.append(np.sum(ewgt_idx_per_layer / event_energy))
        
        return event_lcentroid

    def calculate_r2(self, layers):
        return really_calculate_r2(layers, self.max_events)

    def calculate_lambda2(self, layers):
        return really_calculate_lambda2(layers, self.max_events)

    def calculate_Rz(self, layers):
        event_Rz = []
        # loop over variant events
        for event in range(self.max_events):
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

        return event_Rz

    def calculate_Rx(self, layers):
        event_Rx = []
        # loop over variant events
        for event in range(self.max_events):
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

        return event_Rx

    def make_plot(self,
                  func,
                  histBins,
                  xRange,
                  xLabel,
                  savename):

        # calculate observable
        nominal = func(self.nominal_layers)
        alternative = func(self.layers)
        histos = [nominal, alternative]

        # plot auxiliaries
        labels = ["Nominal", "Alternative", "Corrected"]
        colors=['k', 'r', 'b']
        custom_lines = [Line2D([0], [0], color=color, lw=2) for color in colors[:2]]
        custom_lines.append(Line2D([0], [0], color=colors[2], lw=2, linestyle='--'))
        xMin, xMax = xRange

        # First get the raw counts so that we calculate the uncertainty.
        nsRaw, _, _ = plt.hist(histos, bins=histBins)
        num = 1.0
        denom = np.sqrt(nsRaw)
        relUncs = np.divide(num, denom, out=np.zeros_like(denom), where=denom!=0)

        # make histograms
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, 
                                       gridspec_kw = {'height_ratios':[3, 1]},
                                       sharex=True,
                                       dpi=200) #nrows=2, constrained_layout=True, figsize=(5 , 6)

        ns, bins, patches = ax1.hist(histos, 
                                     bins=histBins,
                                     range=(xMin, xMax),
                                     histtype='step',
                                     label=labels[:2],
                                     linewidth=2,
                                     weights=[100*np.ones((len(histo)))/len((histo)) for histo in histos],
                                     color=colors[:2])

        ns_wgt, bins_wgt, patches_wgt = ax1.hist(alternative,
                                                 weights=100.*self.weights/self.weights.sum(),
                                                 bins=histBins,
                                                 range=(xMin, xMax),
                                                 histtype='step',
                                                 linewidth=2,
                                                 color=colors[2],
                                                 linestyle='--',
                                                 label=labels[2])

        tempbins = np.digitize(np.array(alternative), bins_wgt)
        relWgtUncs = []
        for binI in range(histBins):
            bin_ws = self.weights[np.where(tempbins==binI+1)[0]]
            relWgtUncs.append(np.sqrt(np.sum(bin_ws**2.))/np.sum(bin_ws))
        relWgtUncs = np.array(relWgtUncs)

        # method to write the comparison metrics to the figure
        self.write_metrics(ax1, ns, ns_wgt, histos)

        # add legend
        ax1.legend(custom_lines, labels, loc=2)
        
        # configure main panel axes
        ax1.set_ylabel('Percent of total')
        ax1.set_xlim((xMin, xMax))
        ax1.set_ylim([0, ax1.get_ylim()[1]*1.6])

        # ratio plot
        num = ns[1]
        denom = ns[0]
        binWidth = (xMax-xMin)/histBins
        ratios = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
        ax2.errorbar(bins[:-1]+binWidth/2,     # this is what makes it comparable
                     ratios,
                     linestyle='None',
                     color=colors[1],
                     marker = 'o',
                     yerr=ratios*np.sqrt(np.power(relUncs[0],2)+np.power(relUncs[1],2)),
                     markersize=5)
        num_wgt = ns_wgt
        ratios_wgt = np.divide(num_wgt, denom, out=np.zeros_like(num_wgt), where=denom!=0)
        weightedRelUnc = num_wgt
        ax2.errorbar(bins[:-1]+binWidth/2,     # this is what makes it comparable
                     ratios_wgt,
                     linestyle='None',
                     color=colors[2],
                     marker = 'o',
                     yerr=ratios*np.sqrt(np.power(relUncs[0],2)+np.power(relWgtUncs,2)),
                     markersize=5)
        
        # configure ratio panel axes
        ax2.set_ylabel('Ratio\n(X/Nominal)')
        ax2.set_xlabel(xLabel)
        ax2.set_xlim((xMin, xMax))

        # hline
        ax2.axhline(y=1.0, 
                    color='gray', 
                    linestyle='-',
                    linewidth=0.5)
        
        ax2.set_ylim([0.3, 1.7])
        
        # grid
        ax2.grid(which='major', axis='y')
        fig.subplots_adjust(hspace=0.1)
        fig.canvas.draw()

        # save
        plt.savefig(self.saveDir+f'/{savename}.png', bbox_inches='tight')
        plt.savefig(self.saveDir+f'/{savename}.svg', bbox_inches='tight')
        plt.savefig(self.saveDir+f'/{savename}.pdf', bbox_inches='tight')

        # done
        print("Plotter\t::\tDone plotting %s" % savename)

    def plot_event_observables(self, suffix: str=''):
        '''
        Wrapper function to plot multiple event observables
        '''
        print("Plotter\t::\tPlotting event observables")

        # dict of functions and parameters for different event obsrvables
        observables_config = {'energy_deposit'          : {'func': self.calculate_edep, 'histBins': 20, 'xRange': (150, 230), 'xLabel': 'Energy [MeV]', 'savename': 'edep'+suffix},
                            #   'sparcity'                : {'func': self.calculate_non_zero, 'histBins': 25, 'xRange': (0.0005, 0.0035), 'xLabel': 'Non-zero [%]', 'savename': 'sparcity'+suffix},
                            #   'longitudinal_centroid'   : {'func': self.calculate_longitudinal_centroid, 'histBins': 10, 'xRange': (9, 19), 'xLabel': 'Cell Idx', 'savename': 'l_centroid'+suffix},
                            #   'shower_shape_r2'         : {'func': self.calculate_r2, 'histBins': 30, 'xRange': (300, 600), 'xLabel': 'r2', 'savename': 'r2'+suffix},
                            #   'shower_shape_Rz'         : {'func': self.calculate_Rz, 'histBins': 20, 'xRange': (0.25, 1.25), 'xLabel': 'R_z', 'savename': 'Rz'+suffix},
                            #   'shower_shape_Rx'         : {'func': self.calculate_Rx, 'histBins': 20, 'xRange': (0, 0.5), 'xLabel': 'R_x', 'savename': 'Rx'+suffix},
                              'shower_shape_l2'         : {'func': self.calculate_lambda2, 'histBins': 20, 'xRange': (0, 400), 'xLabel': 'l2', 'savename': 'l2'+suffix}
                             }
        
        for observable in observables_config:
            self.make_plot(**observables_config[observable])

    def write_metrics(self, ax1, ns, ns_wgt, histograms):
        '''
        Bin counts:
        ns[0]: nominal
        ns[1]: alternative
        ns_wgt: alternative*weight

        List of observable values per event
        histograms = [nom_array, alt_array]
        '''
        # distabce metrics
        w_distance = wasserstein_distance(ns[1], ns[0])
        w_distance_wgt = wasserstein_distance(ns_wgt, ns[0])
        js_distance = jensenshannon(ns[1], ns[0])
        js_distance_wgt = jensenshannon(ns_wgt, ns[0])
        # weighted mean standard error of the mean (alternative*weights)
        wmean,werr = wmom(histograms[1], self.weights, inputmean=None, calcerr=True, sdev=False)
        # standard error of the mean (nominal)
        err = sem(histograms[0])
        # ratio = statistical dilution
        r = werr/err

        font_size = 11
        x_left = 0.7
        y_top = 0.92
        y_spacing = 0.08
        
        ax1.text(x_left, y_top, 'WD (Alt.): %.2f' % round(w_distance, 2), transform=ax1.transAxes, fontsize=font_size)
        ax1.text(x_left, y_top-y_spacing, 'WD (Corr.): %.2f' % round(w_distance_wgt, 2), transform=ax1.transAxes, fontsize=font_size)
        ax1.text(x_left, y_top-2*y_spacing, 'JSD (Alt.): %.2f' % round(js_distance, 2), transform=ax1.transAxes, fontsize=font_size)
        ax1.text(x_left, y_top-3*y_spacing, 'JSD (Corr.): %.2f' % round(js_distance_wgt, 2), transform=ax1.transAxes, fontsize=font_size)
        ax1.text(x_left, y_top-4*y_spacing, 'r: %.2f' % round(r, 2), transform=ax1.transAxes, fontsize=font_size)

'''
-----------------
Ploting Functions
-----------------
'''

def plot_calibration_curve(labels, probs: np.ndarray) -> None:
    ''' 
    Plot calibration curve for model 
    '''
    print("Plotter\t::\tPlotting calibration curve")

    fig = plt.figure(figsize=(5 , 5), dpi=200)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    frac_of_pos, mean_pred_value = calibration_curve(labels, probs, n_bins=10)

    ax1.plot(mean_pred_value, frac_of_pos, "s-", label='3DConv')
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    
    ax2.hist(probs, range=(0, 1), bins=10, histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")

    saveDir = 'plots/'+today
    system('mkdir -p '+saveDir)
    plt.savefig(saveDir+'/calibration_curve.png', bbox_inches='tight')

def plot_weights(weights: np.ndarray, suffix: str = '') -> None:
    ''' 
    Plot weights distribution
    '''
    print("Plotter\t::\tPlotting weights")

    plt.figure(figsize=(5 , 5), dpi=200)
    # bins = 10**(np.arange(0,6))
    plt.hist(weights, bins=100, lw=2)
    plt.ylabel("Events")
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel("Weight")

    saveDir = 'plots/'+today
    system('mkdir -p '+saveDir)
    plt.savefig(saveDir+f'/weights{suffix}.png', bbox_inches='tight')
    plt.savefig(saveDir+f'/weights{suffix}.pdf', bbox_inches='tight')
    plt.savefig(saveDir+f'/weights{suffix}.svg', bbox_inches='tight')

def plot_metrics(csvLoggerPath: str, suffix: str = '') -> None:
    ''' 
    Plot metrics such as loss and accuracy.
    '''
    print("Plotter\t::\tPlotting losses")

    # Load CSV file from logger
    metrics = pd.read_csv(csvLoggerPath)
    
    plt.figure(figsize=(5 , 5), dpi=200)
    trainInfo = metrics[metrics['train_loss'].notnull()]
    valInfo = metrics[metrics['val_loss'].notnull()]
    print(trainInfo['step'].max())
    print(valInfo['step'].max())
    plt.plot(trainInfo['step'], trainInfo['train_loss'], label='Train loss')
    plt.plot(valInfo['step'], valInfo['val_loss'], label='Val loss')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    saveDir = 'plots/'+today
    system('mkdir -p '+saveDir)
    plt.savefig(saveDir+f'/loss{suffix}.png', bbox_inches='tight')
    plt.savefig(saveDir+f'/loss{suffix}.pdf', bbox_inches='tight')
    plt.savefig(saveDir+f'/loss{suffix}.svg', bbox_inches='tight')

    plt.figure(figsize=(5 , 5), dpi=200)
    trainInfo = metrics[metrics['train_accuracy'].notnull()]
    valInfo = metrics[metrics['val_accuracy'].notnull()]
    plt.plot(trainInfo['epoch'], trainInfo['train_accuracy'], label='Train loss')
    plt.plot(valInfo['epoch'], valInfo['val_accuracy'], label='Val loss')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    saveDir = 'plots/'+today
    system('mkdir -p '+saveDir)
    plt.savefig(saveDir+f'/accuracy{suffix}.png', bbox_inches='tight')
    plt.savefig(saveDir+f'/accuracy{suffix}.pdf', bbox_inches='tight')
    plt.savefig(saveDir+f'/accuracy{suffix}.svg', bbox_inches='tight')
