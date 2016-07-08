import scipy.io
import numpy as np
import math
from scipy.optimize import fmin_cg
import datetime
import random

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

def flatten_x(x):
	return x.flatten().reshape(training_examples*input_units,1)

def reshape_x(x):
	return x.reshape(training_examples,input_units)

def flatten_thetas(theta1,theta2):
	theta1 = theta1.flatten().reshape(hidden_units*(input_units+1),1)
	theta2 = theta2.flatten().reshape(output_units*(hidden_units+1),1)
	thetas = np.zeros((theta1.size+theta2.size,1))
	thetas[:theta1.size] = theta1
	thetas[theta1.size:] = theta2
	return thetas

def reshape_thetas(thetas):
	theta1 = thetas[:hidden_units*(input_units+1)].reshape(hidden_units,input_units+1)
	theta2 = thetas[hidden_units*(input_units+1):].reshape(output_units,hidden_units+1)
	return theta1,theta2

def gen_random_thetas():
	epsilon = 0.1
	theta1 = np.random.rand(hidden_units,input_units+1).astype(np.float64)*2*epsilon - epsilon
	theta2 = np.random.rand(output_units,hidden_units+1).astype(np.float64)*2*epsilon - epsilon
	return theta1,theta2

def cost(thetas_f,x_f,y,lamda = 0):
	theta1,theta2 = reshape_thetas(thetas_f)
	x = reshape_x(x_f)
	total_cost = 0
	m = x[:,0].size
	
	for i in range(0,m):
		cur_x = x[i,:]
		cur_h = h(theta1,theta2,cur_x)[2][1]
		y_temp = np.zeros((output_units,1))
		y_temp[y[i][0]-1][0]=1
		cost = -y_temp.T.dot(np.log(cur_h))-(1-y_temp.T).dot(np.log(1-cur_h))
		total_cost +=cost
	total_cost = float(total_cost)/m
	
	theta1_temp = np.delete(theta1,0,axis=1)**2
	theta2_temp = np.delete(theta2,0,axis=1)**2
	total_regularized = (theta1_temp.sum()+theta2_temp.sum())*(float(lamda)/(2*m))
	
	return total_cost+total_regularized

def backprop(thetas_f,x_f,y,lamda = 0):
	theta1,theta2 = reshape_thetas(thetas_f)
	x = reshape_x(x_f)
	m = x[:,0].size
	Delta1 = np.zeros((hidden_units,input_units+1))
	Delta2 = np.zeros((output_units,hidden_units+1))
	for i in range(0,m):
		cur_x = x[[i],:]
		cur_h = h(theta1,theta2,cur_x)
		a3 = cur_h[2][1]
		a1 = cur_h[0][0].T
		a2 = cur_h[1][1]
		y_temp = np.zeros((output_units,1))
		y_temp[y[i][0]-1][0]=1
		delta3 = a3-y_temp
		delta2 = theta2.T.dot(delta3)*(a2*(1-a2))
		delta2 = delta2[1:].reshape(hidden_units,1)
		Delta1 += delta2.dot(a1)
		Delta2 += delta3.dot(a2.T)
	D1 = Delta1/float(m)
	D2 = Delta2/float(m)
	
	D1[:,1:] += (float(lamda)/m )*theta1[:,1:]
	D2[:,1:] += (float(lamda)/m )*theta2[:,1:]
	return flatten_thetas(D1,D2).flatten()

def predict(x,y,rval):
	theta1,theta2 = reshape_thetas(rval)
	m = x[:,0].size
	print "number of testing examples = ",m
	count = 0
	for i in range(0,m):
		actual = y[i][0]
		hypothesis =  h(theta1,theta2,x[i])[2][1]
		prediction =  (hypothesis.argmax()+1)
		if actual == prediction:
			count+=1
	print "number of correct predictions = ",count
	print "Accuracy = ",(count*100.0)/m,"%"

def save(rval):
	fname = datetime.datetime.now().strftime("%d-%H:%M:%S")
	np.save(fname,rval)

def preprocess_data(x,y):
	test = np.array(random.sample(range(0,5000),testing_examples))
	train = np.arange(0,5000)
	for i in range(0,testing_examples):
		train[test[i]] = -1
	train = train[train!=-1]
	x_train = np.zeros((training_examples,400))
	y_train = np.zeros((training_examples,400))
	x_test = np.zeros((testing_examples,400))
	y_test = np.zeros((testing_examples,400))
	for i in range(0,training_examples):
		x_train[i] = x[train[i]]
		y_train[i] = y[train[i]]
	for i in range(0,testing_examples):
		x_test[i] = x[test[i]]
		y_test[i] = y[test[i]]
	return x_train,y_train,x_test,y_test

mat1 = scipy.io.loadmat('data.mat')
x_read = mat1['X']
y_read = mat1['y']
input_units = 400
hidden_units = 25
output_units = 10
training_examples = 4500#number of training examples
testing_examples = 5000-training_examples
lamda = 1.0#regularization factor lambda

x_train,y_train,x_test,y_test = preprocess_data(x_read,y_read)

#mat2 = scipy.io.loadmat('weights.mat')
#theta1 = mat2['Theta1']
#theta2 = mat2['Theta2']
#cost(flatten_thetas(theta1,theta2),flatten_x(x),y,lamda = 1)#cost function test
#flattenedD1D2 = backprop(flatten_thetas(theta1,theta2),flatten_x(x),y,lamda = 1)#gradient function test

theta1,theta2 = gen_random_thetas()
x_f = flatten_x(x_train)
thetas_f = flatten_thetas(theta1,theta2)
rval = fmin_cg(cost,thetas_f,fprime=backprop,args=(x_f,y_train,lamda),maxiter=50)

#rval = np.load("08-18:13:36.npy")#to load a backup of trained values

predict(x_test,y_test,rval)

#save(rval)#save the backup of trained values
