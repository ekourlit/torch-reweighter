import numpy as np
from matplotlib import pyplot as plt
import h5py
from datetime import date
today = str(date.today())
from os import system
import pdb
from sklearn.calibration import calibration_curve
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
plt.style.use('default')
font = {'size':14}
matplotlib.rc('font', **font)

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

    def plot_event_edep_WH(self, suffix:str=''):
        '''
        Distribution of event energy deposit
        '''
        print("Plotter\t::\tPlotting event energy deposit")
        colors=['k', 'r', 'b', 'g', 'orange']#, 'c']
        markerstyles=['.' for c in colors]
        custom_lines = [Line2D([0], [0], color=color, lw=2) for color in colors[:2]]
        custom_lines.append(Line2D([0], [0], color=colors[2], lw=2, linestyle='--'))
        nom_edep = self.calculate_edep(self.nominal_layers)
        alt_edep = self.calculate_edep(self.layers)
        histos = [nom_edep, alt_edep]
        
        #fig.suptitle('Event energy deposit')
        myBins = 20
        xmin = 150
        xmax = 230.
        binWidth = (xmax-xmin)/myBins
        density = True
        #labels = ["Nominal", "Alternative", "Alternative*Weight"]
        labels = ["Nominal (0.1 mm)", "10 mm", "Corrected 10 mm"]

        # First get the raw counts so that we calculate the uncertainty.
        nsRaw, bins, patches = plt.hist(histos, bins=myBins)
        relUncs = 1./np.sqrt(nsRaw)
        # plot histo
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]}, sharex=True, dpi=200)#(nrows=2, constrained_layout=True, figsize=(5 , 6), dpi=200)
        ns, bins, patches = ax1.hist(histos, 
                                     bins=myBins,
                                     range=(xmin, xmax),
                                     histtype='step',
                                     #alpha=0.4,
                                     label=labels[:2],
                                     linewidth=2,
                                     #density=density,
                                     weights=[100*np.ones((len(histo)))/len((histo)) for histo in histos],
                                     color=colors[:2])

        ns_wgt, bins_wgt, patches_wgt = ax1.hist(alt_edep,
                                                 weights=100*self.weights/self.weights.sum(),
                                                 bins=myBins,
                                                 range=(xmin, xmax),
                                                 histtype='step',
                                                 linewidth=2,
                                                 color=colors[2],
                                                 linestyle='--',
                                                 label=labels[2],
                                                 #density=density
                                                 )

        w_distance = wasserstein_distance(ns[1], ns[0])
        w_distance_wgt = wasserstein_distance(ns_wgt, ns[0])
        js_distance = jensenshannon(ns[1], ns[0])
        js_distance_wgt = jensenshannon(ns_wgt, ns[0])
        font = 8
        # ax1.text(0.05, 0.86, 'WD (Alternative): %.4f' % w_distance, transform=ax1.transAxes, fontsize=font)
        # ax1.text(0.05, 0.80, 'WD (Alternative*Weight): %.4f' % w_distance_wgt, transform=ax1.transAxes, fontsize=font)
        # ax1.text(0.05, 0.74, 'JSD (Alternative): %.4f' % js_distance, transform=ax1.transAxes, fontsize=font)
        # ax1.text(0.05, 0.68, 'JSD (Alternative*Weight): %.4f' % js_distance_wgt, transform=ax1.transAxes, fontsize=font)

        ax1.legend(custom_lines, labels, loc=2)
        #ax1.set_ylabel('Events')
        ax1.set_ylabel('Percent of total')
        ax1.set_xlim((xmin, xmax))
        ax1.set_ylim([0, ax1.get_ylim()[1]*1.6])

        # ratio plot
        num = ns[1]
        denom = ns[0]
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
        ax2.errorbar(bins[:-1]+binWidth/2,     # this is what makes it comparable
                ratios_wgt,
                linestyle='None',
                     color=colors[2],
                marker = 'o',
                markersize=5)

        ax2.set_ylabel('Ratio\n(X/Nominal)')
        ax2.set_xlabel('Energy [MeV]')
        ax2.set_xlim((xmin, xmax))

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
        plt.savefig(self.saveDir+f'/edep{suffix}.png', bbox_inches='tight')
        plt.savefig(self.saveDir+f'/edep{suffix}.svg', bbox_inches='tight')
        plt.savefig(self.saveDir+f'/edep{suffix}.pdf', bbox_inches='tight')


    def plot_event_edep(self, suffix:str =''):
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

        myBins = 20
        xmin = 100
        xmax = 300
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

        w_distance = wasserstein_distance(ns[1], ns[0])
        w_distance_wgt = wasserstein_distance(ns_wgt, ns[0])
        js_distance = jensenshannon(ns[1], ns[0])
        js_distance_wgt = jensenshannon(ns_wgt, ns[0])
        font = 6
        ax1.text(0.05, 0.86, 'WD (Alternative): %.4f' % w_distance, transform=ax1.transAxes, fontsize=font)
        ax1.text(0.05, 0.80, 'WD (Alternative*Weight): %.4f' % w_distance_wgt, transform=ax1.transAxes, fontsize=font)
        ax1.text(0.05, 0.74, 'JSD (Alternative): %.4f' % js_distance, transform=ax1.transAxes, fontsize=font)
        ax1.text(0.05, 0.68, 'JSD (Alternative*Weight): %.4f' % js_distance_wgt, transform=ax1.transAxes, fontsize=font)

        ax1.legend()
        ax1.set_ylabel('Events')
        ax1.set_xlim((xmin, xmax))

        # ratio plot
        num = ns[1]
        denom = ns[0]
        ratios = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
        ax2.errorbar(bins[:-1],     # this is what makes it comparable
                ratios,
                linestyle='None',
                color='C1',
                marker = 'o',
                markersize=5)
        num_wgt = ns_wgt
        ratios_wgt = np.divide(num_wgt, denom, out=np.zeros_like(num_wgt), where=denom!=0)
        ax2.errorbar(bins[:-1],     # this is what makes it comparable
                ratios_wgt,
                linestyle='None',
                color='k',
                marker = 'o',
                markersize=5)

        ax2.set_ylabel('Ratio (Alt./Nom.)')
        ax2.set_xlabel('Energy [MeV]')
        ax2.set_xlim((xmin, xmax))

        # hline
        ax2.axhline(y=1.0, 
                    color='gray', 
                    linestyle='-',
                    linewidth=0.5)
        ax2.set_ylim([0.5, 2])
        # grid
        ax2.grid(which='major', axis='y')
        
        plt.savefig(self.saveDir+f'/edep{suffix}.png', bbox_inches='tight')
        
    def plot_event_sparcity(self, suffix:str =''):
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
        
        plt.savefig(self.saveDir+f'/cell_sparcity{suffix}.png', bbox_inches='tight')

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
