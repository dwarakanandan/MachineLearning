import numpy as np
import math
from scipy.misc import imread
from skimage import img_as_float

def g(z):
	return 1/(1+math.e**(-(z)))

def h(theta1,theta2,x):
	a1 = np.ones((input_units+1,1))
	a1[1:]=x.reshape(input_units,1)
	z2 = theta1.dot(a1)
	a2 = np.ones((hidden_units+1,1))
	a2[1:] = g(z2)
	z3 = theta2.dot(a2)
	a3 = g(z3)
	return [[a1],[z2,a2],[z3,a3]]

def reshape_thetas(thetas):
	theta1 = thetas[:hidden_units*(input_units+1)].reshape(hidden_units,input_units+1)
	theta2 = thetas[hidden_units*(input_units+1):].reshape(output_units,hidden_units+1)
	return theta1,theta2

def predict(x,rval):
	theta1,theta2 = reshape_thetas(rval)
	hypothesis =  h(theta1,theta2,x)[2][1]
	prediction =  (hypothesis.argmax()+1)
	print "This is ",prediction

np.seterr(all="ignore")
input_units = 784
hidden_units = 300
output_units = 10
rval = np.load("/home/nandu/Desktop/Machine learning/Neural network (MINST dataset)/results/2L-NN-784-300.npy")
x_test = imread("p1.png")
y_test= 0
predict(x_test.flatten(),rval)
