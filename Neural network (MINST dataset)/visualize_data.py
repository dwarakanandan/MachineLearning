#randomly select 25 images from the data set and stitch them to make a big image
import numpy as np
import matplotlib.pyplot as plt
import random
import minst

x,y = minst.load_mnist("train")
m =  x[:,0,0].size
x =  x.reshape(m,784)

a = np.array([random.randint(0,m) for i in range(0,25)]).reshape(5,5)
img = np.zeros([5,5,28,28])
label = np.ones([5,5])
for i in range(0,5):
	for j in range(0,5):
		img[i][j] = x[a[i][j]].reshape(28,28)
		label[i][j] = y[a[i][j]]
print label
big_pic = np.zeros((140,140))
for i in range(0,5):
	for j in range(0,5):
		for k in range(0,28):
			for l in range(0,28):
				big_pic[i*28+k][j*28+l] = img[i][j][k][l]

print x[0]
plt.imshow(big_pic,cmap="Greys_r")#display the big image
plt.show()
