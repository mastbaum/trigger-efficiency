import numpy as np

def load(filename,
         xfilt = lambda x: True,
         yfilt = lambda x: True,
         zfilt = lambda x: True):
    '''Read a trigger scan data file in as a numpy array, applying filter
    functions to the x, y, and z coordinates.
    '''
    data = np.loadtxt(filename)

    # set data range limits
    xfiltv = np.vectorize(xfilt)
    yfiltv = np.vectorize(yfilt)
    zfiltv = np.vectorize(zfilt)

    # apply limits to data
    data = data[xfiltv(data[:,0]) & yfiltv(data[:,1]) & zfiltv(data[:,2])]

    return data

