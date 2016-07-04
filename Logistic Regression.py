import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt

def show_scatter():
	data_admitted = data[data[:,2]==1]
	data_notadmitted = data[data[:,2]==0]
	plt.scatter(data_admitted[:,0],data_admitted[:,1],c='r',s=50)
	plt.scatter(data_notadmitted[:,0],data_notadmitted[:,1],c='b',s=50)
	x_coordinates = [0,-theta_ans[0][0]/theta_ans[1][0]]
	y_coordinates = [-theta_ans[0][0]/theta_ans[2][0],0]
	plt.plot(x_coordinates,y_coordinates)
	plt.show()

def h(x,theta):
	return 1/(1+math.e**(-(x.dot(theta))))

def cost(theta):
	hypothesis = h(x,theta)
	error = -y.dot(np.log(hypothesis.clip(min=0.00000001)))-(1-y).dot(np.log((1-hypothesis).clip(min=0.00000001)))
	return error.sum()/m

def gradient(theta):
	grad = np.zeros([3,1])
	hypothesis = h(x,theta).flatten()
	for i in range(0,3):
		grad[i][0]= ((hypothesis-y).dot(x[:,i]))/m
	return grad.flatten()

def gradient_descent(x,y,theta,alpha):
	for iterations in range(0,1000000):
		hypothesis = h(x,theta).flatten()
		for i in range(0,3):
			theta[i][0]-= alpha * ((hypothesis-y).dot(x[:,i]))/m
	return theta

def predict(theta,data):
	m = data[:,[0]].size
	x1 = data[:,0]
	x2 = data[:,1]
	x = np.ones([m,3])
	x[:,1] = x1
	x[:,2] = x2
	y = data[:,2]
	predictions =  x.dot(theta)
	predictions[predictions[:,0]<0]=0
	predictions[predictions[:,0]>0]=1
	correct=0
	for i in range(0,m):
		if y[i]==predictions[i]:
			correct+=1
	print "Accuracy = ",correct*100/m,"%"

data1 = np.loadtxt("data.txt",delimiter=",")
data = data1[0:80]
m = data[:,[0]].size
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]
x = np.ones([m,3])
x[:,1] = x1
x[:,2] = x2
theta = np.zeros([3,1])
ret = opt.fmin_bfgs(cost,theta,fprime=gradient)#inbuilt minimization using bfgs algorithm
theta_ans = np.array([[ret[0]],[ret[1]],[ret[2]]])
#theta_ans = gradient_descent(x,y,theta,0.001)#my gradient descent
print theta_ans
#theta_ans = np.array([[-15.39517],[0.128259],[0.122479]])
#theta_ans = np.array([[-15.66848],[0.13],[0.12266]])
predict(theta_ans,data1[80:])
print theta_ans[0]+45*theta_ans[1]+85*theta_ans[2]
show_scatter()
