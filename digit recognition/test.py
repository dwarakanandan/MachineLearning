import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(suppress=True)
mat = scipy.io.loadmat('data.mat')
a = mat['X']
img = a[[1700],:]
print img
i1 = img.reshape(20,20).T
img = a[[1702],:]
i2 = img.reshape(20,20).T
my = mpimg.imread('pic.png')
b = my.T.reshape(1,400)
b = b.astype(np.float64, copy=False)
print b
