#randomly select 25 images from the data set and stitch them to make a big image
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random

mat1 = scipy.io.loadmat('data.mat')
x = mat1['X']
y = mat1['y']

a = np.array([random.randint(0,5000) for i in range(0,25)]).reshape(5,5)
img = np.zeros([5,5,20,20])
label = np.ones([5,5])
for i in range(0,5):
	for j in range(0,5):
		img[i][j] = x[a[i][j]].reshape(20,20).T
		label[i][j] = y[a[i][j]]
print label
big_pic = np.zeros((100,100))
for i in range(0,5):
	for j in range(0,5):
		for k in range(0,20):
			for l in range(0,20):
				big_pic[i*20+k][j*20+l] = img[i][j][k][l]


plt.imshow(big_pic,cmap="Greys_r")#display the big image
plt.show()
