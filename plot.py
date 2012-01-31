import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        print 'Usage:', sys.argv[0], '[trigger scan file]'
        sys.exit(1)

    data = np.loadtxt(sys.argv[1])

    # set data range limits
    xfilt = np.vectorize(lambda x: True)
    yfilt = np.vectorize(lambda x: x < 50)
    zfilt = np.vectorize(lambda x: x < 1.0e6)

    # apply limits to data
    mask = xfilt(data[:,0])
    mask &= yfilt(data[:,1])
    mask &= zfilt(data[:,2])
    data = data[mask]

    plot_trigscan(data)

