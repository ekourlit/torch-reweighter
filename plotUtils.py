import numpy as np
from matplotlib import pyplot as plt
import h5py
from datetime import date
today = str(date.today())
from os import system
import pdb
from sklearn.calibration import calibration_curve

class Plotter:
    """
    Class to create plots. 
    The different kind of plots are created by the class methods.
    """

    def __init__(self, nominal_dataset: h5py._hl.group.Group, dataset: h5py._hl.group.Group, weights: np.ndarray) -> None:
        self.max_events = 4900
        # construct np.arrays
        self.nominal_layers = nominal_dataset['layers'][:][:self.max_events, :, :, :] # shape: (self.max_events, 30, 30, 30)
        self.layers = dataset['layers'][:][:self.max_events, :, :, :] # shape: (self.max_events, 30, 30, 30)
        self.weights = weights[:self.max_events] # shape: (self.max_events, )

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

    def plot_event_edep(self):
        '''
        Distribution of event energy deposit
        '''
        print("Plotter\t::\tPlotting event energy deposit")

        nom_edep = self.calculate_edep(self.nominal_layers)
        alt_edep = self.calculate_edep(self.layers)
        histos = [nom_edep, alt_edep]

        # plot histo
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(nrows=2, constrained_layout=True, figsize=(5 , 6), dpi=200)
        fig.suptitle('Event energy deposit')

        myBins = 25
        xmin = 0
        xmax = 50
        density = True

        ns, bins, patches = ax1.hist(histos, 
                                     bins=myBins,
                                     range=(xmin, xmax),
                                     histtype='stepfilled',
                                     alpha=0.4,
                                     label=["Nominal", "Alternative"],
                                     density=density)

        ns_wgt, bins_wgt, patches_wgt = ax1.hist(alt_edep,
                                                 weights=self.weights,
                                                 bins=myBins,
                                                 range=(xmin, xmax),
                                                 histtype='step',
                                                 linewidth=1,
                                                 color='k',
                                                 linestyle='--',
                                                 label="Alternative*Weight",
                                                 density=density)

        ax1.legend()
        ax1.set_ylabel('Events')

        # ratio plot
        ax2.bar(bins[:-1],     # this is what makes it comparable
                np.divide(ns[1], ns[0], out=np.zeros_like(ns[1]), where=ns[0]!=0),
                alpha=0.4,
                color='C1')
        ax2.bar(bins[:-1],     # this is what makes it comparable
                np.divide(ns_wgt, ns[0], out=np.zeros_like(ns_wgt), where=ns[0]!=0),
                fill=False,
                linewidth=1,
                color='k',
                linestyle='--')

        # hline
        ax2.axhline(y=1.0, 
                    color='r', 
                    linestyle='-',
                    linewidth=0.5)

        ax2.set_ylim([0, 4])
        ax2.set_ylabel('Ratio (Alt./Nom.)')
        ax2.set_xlabel('Energy [MeV]')
        
        plt.savefig(self.saveDir+'/edep.png', bbox_inches='tight')

    def plot_event_sparcity(self):
        '''
        Distribution of event cell sparcity
        vangelis: something is going wrong with the bottom panel
        '''
        print("Plotter\t::\tPlotting event cell sparcity")

        nom_edep = self.calculate_non_zero(self.nominal_layers)
        alt_edep = self.calculate_non_zero(self.layers)
        histos = [nom_edep, alt_edep]

        # plot histo
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(nrows=2, constrained_layout=True, figsize=(5 , 6), dpi=200)
        fig.suptitle('Event cell sparcity')

        myBins = 25
        xmin = 0.0005
        xmax = 0.0035
        density = True

        ns, bins, patches = ax1.hist(histos, 
                                     bins=myBins,
                                     range=(xmin, xmax),
                                     histtype='stepfilled',
                                     alpha=0.4,
                                     label=["Nominal", "Alternative"],
                                     density=density)

        ns_wgt, bins_wgt, patches_wgt = ax1.hist(alt_edep,
                                                 weights=self.weights,
                                                 bins=myBins,
                                                 range=(xmin, xmax),
                                                 histtype='step',
                                                 linewidth=1,
                                                 color='k',
                                                 linestyle='--',
                                                 label="Alternative*Weight",
                                                 density=density)

        ax1.legend()
        ax1.set_ylabel('Events')

        # ratio plot
        ax2.bar(bins[:-1],     # this is what makes it comparable
                np.divide(ns[1], ns[0], out=np.zeros_like(ns[1]), where=ns[0]!=0),
                alpha=0.4,
                color='C1')
        ax2.bar(bins[:-1],     # this is what makes it comparable
                np.divide(ns_wgt, ns[0], out=np.zeros_like(ns_wgt), where=ns[0]!=0),
                fill=False,
                linewidth=1,
                color='k',
                linestyle='--')

        # hline
        ax2.axhline(y=1.0, 
                    color='r', 
                    linestyle='-',
                    linewidth=0.5)

        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim([0, 4])
        ax2.set_ylabel('Ratio (Alt./Nom.)')
        ax2.set_xlabel('Cell Sparcity')
        
        plt.savefig(self.saveDir+'/cell_sparcity.png', bbox_inches='tight')

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

def plot_weights(weights: np.ndarray) -> None:
    ''' 
    Plot weights distribution
    '''
    print("Plotter\t::\tPlotting weights")

    plt.figure(figsize=(5 , 5), dpi=200)
    plt.hist(weights, bins=100, lw=2)
    plt.ylabel("Events")
    plt.yscale('log')    
    plt.xlabel("Weight")

    saveDir = 'plots/'+today
    system('mkdir -p '+saveDir)
    plt.savefig(saveDir+'/weights.png', bbox_inches='tight')