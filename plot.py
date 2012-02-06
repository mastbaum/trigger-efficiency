import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tools

def plot_trigscan(data):
    '''Plot a trigger scan data set in 3D. `data` is a numpy.array with
    shape=(npoints,3).
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:,0], data[:,1], np.log10(data[:,2]), c='r', marker='+')
    ax.set_xlabel('DAC')
    ax.set_ylabel('Channels')
    ax.set_zlabel('Log efficiency')

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage:', sys.argv[0], '<trigger scan file> [start_nhit end_nhit]'
        sys.exit(1)

    if len(sys.argv) == 4:
        xfilt = lambda x: x > float(sys.argv[2]) and x < float(sys.argv[3])
    else:
        xfilt = lambda x: True

    zfilt = lambda x: x < 1.0e6 and x > 0

    data = tools.load(sys.argv[1], xfilt=xfilt, zfilt=zfilt)

    print data
    plot_trigscan(data)

