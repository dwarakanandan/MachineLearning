import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt

def h(x,theta):
	return 1/(1+math.e**(-(x.dot(theta))))

def cost(theta):
	hypothesis = h(x,theta)
	error = -y.dot(np.log(hypothesis.clip(min=0.00000001)))-(1-y).dot(np.log((1-hypothesis).clip(min=0.00000001)))+lam*((theta**2).sum()-theta[0]**2)/2*m
	return error.sum()/m

def gradient(theta):
	grad = np.zeros([num_features,1])
	hypothesis = h(x,theta).flatten()
	grad[0][0]= ((hypothesis-y).dot(x[:,0]))/m
	for i in range(1,num_features):
		grad[i][0]= (((hypothesis-y).dot(x[:,i]))/m)+((lam*theta[i])/m)
	return grad.flatten()

def map_features(x1,x2,m):
	x = np.ones([m,num_features])
	x[:,1] = x1
	x[:,2] = x2
	x[:,3] = x1**2
	x[:,4] = x1*x2
	x[:,5] = x2**2
	x[:,6] = x1**3
	x[:,7] = (x1**2)*x2
	x[:,8] = x1*(x2**2)
	x[:,9] = x2**3
	x[:,10] = x1**4
	x[:,11] = (x1**3)*x2
	x[:,12] = (x1**2)*(x2**2)
	x[:,13] = x1*(x2**3)
	x[:,14] = x2**4
	x[:,15] = x1**5
	x[:,16] = (x1**4)*x2
	x[:,17] = (x1**3)*(x2**2)
	x[:,18] = (x1**2)*(x2**3)
	x[:,19] = x1*(x2**4)
	x[:,20] = x2**5
	x[:,21] = x1**6
	x[:,22] = (x1**5)*x2
	x[:,23] = (x1**4)*(x2**2)
	x[:,24] = (x1**3)*(x2**3)
	x[:,25] = (x1**2)*(x2**4)
	x[:,26] = x1*(x2**5)
	x[:,27] = x2**6
	return x.clip(min=0.0000000001)
	
def predict(theta,data):
	m = data[:,0].size
	x1 = data[:,0]
	x2 = data[:,1]
	y = data[:,2]
	X = map_features(x1,x2,m)
	predictions =  X.dot(theta)
	predictions[predictions[:,0]<0]=0
	predictions[predictions[:,0]>0]=1
	correct=0
	for i in range(0,m):
		if y[i]==predictions[i]:
			correct+=1
	print "Accuracy = ",correct*100/m,"%\n\n"

data1 = np.loadtxt("data_microchip.txt",delimiter=",")
data = data1[0:100]
m = data[:,[0]].size
num_features = 28
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]
theta = np.zeros([num_features,1])
x = map_features(x1,x2,m)

lam = 1
theta_ans = opt.fmin_bfgs(cost,theta,fprime=gradient)#inbuilt minimization using bfgs algorithm
theta_ans = theta_ans.reshape(28,1)
print "Best fit(lambda=1): ",
predict(theta_ans,data1[0:100])

lam = 100
theta_ans = opt.fmin_bfgs(cost,theta,fprime=gradient)
theta_ans = theta_ans.reshape(28,1)
print "Underfitting fitting(lambda=100): ",
predict(theta_ans,data1[0:100])

lam = 0
theta_ans = opt.fmin_bfgs(cost,theta,fprime=gradient)
theta_ans = theta_ans.reshape(28,1)
print "Overfitting fitting(lambda=0): ",
predict(theta_ans,data1[0:100])
