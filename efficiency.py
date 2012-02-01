import sys
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_efficiency(means, stds):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.arange(len(means))

    ax.errorbar(x, means, yerr=stds)

    ax.set_xlim(min(x)-1, max(x)+1)
    ax.set_xlabel('Channels enabled')
    ax.set_ylabel('Trigger efficiency')

    plt.show()

def load(filename,
         xfilt = lambda x: True,
         yfilt = lambda x: True,
         zfilt = lambda x: True):
    '''Read a trigger scan data file in as a numpy array, applying filter
    functions to the x, y, and z coordinates.
    '''
    data = np.loadtxt(sys.argv[1])

    # set data range limits
    xfiltv = np.vectorize(xfilt)
    yfiltv = np.vectorize(yfilt)
    zfiltv = np.vectorize(zfilt)

    # apply limits to data
    data = data[xfiltv(data[:,0]) & yfiltv(data[:,1]) & zfiltv(data[:,2])]

    return data

def make_efficiency_curve(data, plot_overlay=False):
    '''Scans through data (numpy.array, shape=(n,3)) to generate an efficiency
    curve, i.e. a projection of the trigger turn-on profile. Returns a tuple of
    (means for each nchannel, standard deviations for nchannel)
    '''
    data.view('i8,i8,f8').sort(order=['f0','f1'], axis=0)
    groups = [[a[-1] for a in g] for k,g in it.groupby(data, lambda x: x[0])]
    eff = np.array([g[-min(map(len, groups)):] for g in groups])

    means = np.average(eff, axis=0)
    stds = np.std(eff, axis=0)

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

    return means, stds

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage:', sys.argv[0], '<trigger scan file> [start_nhit end_nhit]'
        sys.exit(1)

    if len(sys.argv) == 4:
        xf = lambda x: x > float(sys.argv[2]) and x < float(sys.argv[3])
    else:
        xf = lambda x: True

    zf = lambda x: x < 1.0e6

    data = load(sys.argv[1], xfilt=xf, zfilt=zf)

    plot_efficiency(*make_efficiency_curve(data, plot_overlay=True))

