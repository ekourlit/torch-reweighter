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
from variables import calculate_edep_np, calculate_non_zero_np, calculate_longitudinal_centroid_np, calculate_r2_np, calculate_Rz_np, calculate_Rx_np, calculate_lambda2_np
from pytorch_lightning import Trainer

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

    def make_plot(self,
                  func,
                  histBins,
                  xRange,
                  xLabel,
                  savename):

        # calculate observable
        nominal = func(self.max_events, self.nominal_layers)
        alternative = func(self.max_events, self.layers)
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
            bin_sumOfws = np.sum(bin_ws)
            if bin_sumOfws != 0:
                relWgtUncs.append(np.sqrt(np.sum(bin_ws**2.))/bin_sumOfws)
            else:
                relWgtUncs.append(0.0)
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
        observables_config = {'energy_deposit'          : {'func': calculate_edep_np,                     'histBins': 20, 'xRange': (150, 230),       'xLabel': 'Energy [MeV]',   'savename': 'edep'+suffix},
                              'sparsity'                : {'func': calculate_non_zero_np,                 'histBins': 28, 'xRange': (0.008, 0.015),   'xLabel': 'Non-zero [%]',   'savename': 'sparsity'+suffix},
                              'longitudinal_centroid'   : {'func': calculate_longitudinal_centroid_np,    'histBins': 10, 'xRange': (9, 19),          'xLabel': 'Cell Idx',       'savename': 'l_centroid'+suffix},
                              'shower_shape_r2'         : {'func': calculate_r2_np,                       'histBins': 20, 'xRange': (350, 550),       'xLabel': 'r2',             'savename': 'r2'+suffix},
                              'shower_shape_Rz'         : {'func': calculate_Rz_np,                       'histBins': 20, 'xRange': (0.25, 1.25),     'xLabel': 'R_z',            'savename': 'Rz'+suffix},
                              'shower_shape_Rx'         : {'func': calculate_Rx_np,                       'histBins': 20, 'xRange': (0, 0.5),         'xLabel': 'R_x',            'savename': 'Rx'+suffix},
                              'shower_shape_l2'         : {'func': calculate_lambda2_np,                  'histBins': 20, 'xRange': (0, 400),         'xLabel': 'l2',             'savename': 'l2'+suffix}
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

def plot_training_metrics(trainer: Trainer) -> None:
    metrics = trainer.callbacks[0].metrics
    
    saveDir = 'plots/'+today
    system('mkdir -p '+saveDir)

    fig, ax = plt.subplots()
    ax.plot(metrics['loss'])
    ax.plot(metrics['valid_loss'])
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    plt.savefig(saveDir+f'/loss.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(metrics['accuracy'])
    ax.plot(metrics['valid_accuracy'])
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    plt.savefig(saveDir+f'/accuracy.pdf', bbox_inches='tight')
