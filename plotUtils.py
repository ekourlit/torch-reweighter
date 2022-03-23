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
        labels = ["Nominal", "Alternative", "Corrected"]

        # First get the raw counts so that we calculate the uncertainty.
        nsRaw, bins, patches = plt.hist(histos, bins=myBins)
        relUncs = 1./np.sqrt(nsRaw)
        

        # plot histo
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2, 1, 
                                       gridspec_kw = {'height_ratios':[3, 1]},
                                       sharex=True,
                                       dpi=200) #nrows=2, constrained_layout=True, figsize=(5 , 6)

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
                                                 weights=100.*self.weights/self.weights.sum(),
                                                 bins=myBins,
                                                 range=(xmin, xmax),
                                                 histtype='step',
                                                 linewidth=2,
                                                 color=colors[2],
                                                 linestyle='--',
                                                 label=labels[2],
                                                 #density=density
                                                 )
        
        tempbins = np.digitize(np.array(alt_edep), bins_wgt)
        relWgtUncs = []
        for binI in range(myBins):
            bin_ws = self.weights[np.where(tempbins==binI+1)[0]]
            relWgtUncs.append(np.sqrt(np.sum(bin_ws**2.))/np.sum(bin_ws))
        relWgtUncs = np.array(relWgtUncs)

        # method to write the comparison metrics to the figure
        self.write_metrics(ax1, ns, ns_wgt, histos)

        ax1.legend(custom_lines, labels, loc=2)
        
        if density:
            ax1.set_ylabel('Percent of total')
        else:
            ax1.set_ylabel('Events')

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
        weightedRelUnc = num_wgt
        ax2.errorbar(bins[:-1]+binWidth/2,     # this is what makes it comparable
                     ratios_wgt,
                     linestyle='None',
                     color=colors[2],
                     marker = 'o',
                     yerr=ratios*np.sqrt(np.power(relUncs[0],2)+np.power(relWgtUncs,2)),
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

        # save
        plt.savefig(self.saveDir+f'/edep{suffix}.png', bbox_inches='tight')
        plt.savefig(self.saveDir+f'/edep{suffix}.svg', bbox_inches='tight')
        plt.savefig(self.saveDir+f'/edep{suffix}.pdf', bbox_inches='tight')

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
