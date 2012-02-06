import sys
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.optimize

import tools

def linear_fit_plot(x, y):
    x = np.array(x)
    y = np.array(y)

    line = lambda v, x: v[0] * x + v[1]
    e = lambda v, x, y: ((line(v, x) - y)**2).sum()

    vopt = scipy.optimize.fmin(e, (1,1), args=(x,y))
    yfit = line(vopt, x)

    x = 4096 - x
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_data = ax.plot(x, y)
    plot_fit = ax.plot(x, yfit)

    ax.legend([plot_data, plot_fit], ['Data', 'Linear Fit'])
    ax.set_xlim(max(x)+1, min(x)-1)
    ax.set_xlabel('ADC')
    ax.set_ylabel('NHIT')

    print 'Best fit: v =', vopt
    print 'ADC = 4096 - ( ( NHIT -', vopt[1], ') /', vopt[0] ,')'

    plt.show()

def make_nhit_vs_adc(data, plot_fits=False):
    '''finds the nhit-per-adc function for the data set, using the offset of a
    fit logistic curve.
    '''
    logistic = lambda v,x: 1.0  / (1.0 + np.exp(-v[0]*(x-v[1])))
    e = lambda v,x,y: ((logistic(v,x)-y)**2).sum()

    data.view('i8,i8,f8').sort(order=['f0','f1'], axis=0)

    groups = {}
    for k, g in it.groupby(data, lambda x: x[0]):
        groups[k] = zip(*[(a[1], a[2]) for a in g])

    offsets = {}
    for group in groups:
        x, y = groups[group]

        maxy = max(y)
        if not maxy > 0.999 and maxy < 1.001:
            continue

        try:
            v0 = [1.0, min(x)]
            popt = scipy.optimize.fmin(e, v0, args=(x, y), maxiter=500, maxfun=500)
            offsets[group] = popt[1]

            if plot_fits:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                xx = np.arange(min(x), max(x))
                yy = logistic(popt, xx)
                ax.plot(x, y, xx, yy)
                plt.show()
            
        except RuntimeError:
            offsets[group] = [0]

    return zip(*sorted(offsets.iteritems()))

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

    d = make_nhit_vs_adc(data, False)
    print d
    linear_fit_plot(*d)

    #plot(*make_nhit_vs_adc(data))

