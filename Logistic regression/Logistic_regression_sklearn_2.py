import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr

def map_features(x1,x2,m):
	x = np.ones([m,28])
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
	return x

data = np.loadtxt("data_microchip.txt",delimiter=",")
m = data[:,0].size
x1 = data[:,0]
x2 = data[:,1]
x = map_features(x1,x2,m)
y = data[:,2]

reg = lr(C=10)
reg.fit(x,y)

s = reg.coef_.size
theta_ans = np.zeros((s+1))
theta_ans[0] = reg.intercept_[0]
theta_ans[1:] = reg.coef_
theta_ans = theta_ans.reshape(s+1,1)
print "%.2f%% accuracy"%(reg.score(x,y)*100)
