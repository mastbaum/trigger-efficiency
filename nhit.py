import sys
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.optimize

import tools

def plot_efficiency(means, stds):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.arange(len(means))

    ax.errorbar(x, means, yerr=stds)

    ax.set_xlim(min(x)-1, max(x)+1)
    ax.set_xlabel('Channels enabled')
    ax.set_ylabel('Trigger efficiency')

    plt.show()

def make_nhit_vs_adc(data, plot_overlay=False):
    '''finds the nhit-per-adc function for the data set, using the offset of a
    fit logistic curve.
    '''
    def logistic(x, a, b):
        return 1.0 / (1.0 + np.exp(-a*(x-b)))

    data.view('i8,i8,f8').sort(order=['f0','f1'], axis=0)

    groups = {}
    for k, g in it.groupby(data, lambda x: x[0]):
        groups[k] = zip(*[(a[1], a[2]) for a in g])

    a = {}
    for group in groups:
        x, y = groups[group]

        maxy = max(y)
        if not maxy > 0.999 and maxy < 1.001:
            continue

        try:
            popt, pcov = scipy.optimize.curve_fit(logistic, x, y)
            a[group] = popt
        except RuntimeError:
            a[group] = [0,0]

    for i in a:
        print i, a[i]

    #eff = np.array([g[-min(map(len, groups)):] for g in groups])

    #means = np.average(eff, axis=0)
    #stds = np.std(eff, axis=0)

    if plot_overlay:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.arange(len(eff[0]))
        for g in eff:
            ax.plot(x, g)

        ax.set_xlim(min(x)-1, max(x)+1)
        ax.set_xlabel('Channels enabled')
        ax.set_ylabel('Trigger efficiency')
        plt.show()

    #return means, stds

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage:', sys.argv[0], '<trigger scan file> [start_nhit end_nhit]'
        sys.exit(1)

    if len(sys.argv) == 4:
        xf = lambda x: x > float(sys.argv[2]) and x < float(sys.argv[3])
    else:
        xf = lambda x: True

    zf = lambda x: x < 1.0e6

    data = tools.load(sys.argv[1], xfilt=xf, zfilt=zf)

    make_nhit_vs_adc(data)

    #plot(*make_nhit_vs_adc(data))

