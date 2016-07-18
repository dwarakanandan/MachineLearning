import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr

def show_scatter():
	data_admitted = data[data[:,2]==1]
	data_notadmitted = data[data[:,2]==0]
	plt.scatter(data_admitted[:,0],data_admitted[:,1],c='r',s=50)
	plt.scatter(data_notadmitted[:,0],data_notadmitted[:,1],c='b',s=50)
	x_coordinates = [0,-theta_ans[0][0]/theta_ans[1][0]]
	y_coordinates = [-theta_ans[0][0]/theta_ans[2][0],0]
	plt.plot(x_coordinates,y_coordinates)
	plt.show()


data = np.loadtxt("data_logistic_regression.txt",delimiter=",")
m = data[:,0].size
x = data[:,0:2]
y = data[:,2]

reg = lr(C=3.2)
reg.fit(x,y)
s = reg.coef_.size
theta_ans = np.zeros((s+1))
theta_ans[0] = reg.intercept_[0]
theta_ans[1:] = reg.coef_
theta_ans = theta_ans.reshape(s+1,1)
print theta_ans
print reg.score(x,y)*100,"% accuracy"
show_scatter()
