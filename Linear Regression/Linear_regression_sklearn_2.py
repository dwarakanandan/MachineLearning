import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr

data = np.loadtxt("ex1data2.txt",delimiter = ',')
m = data[:,0].size
x = data[:,0:2]
y = data[:,2]
a = lr(fit_intercept=True)
a.fit(x,y)
print a.coef_
print a.intercept_
print a.score(x,y)
manual = a.intercept_+x.dot(a.coef_.T)#predict manually
for i in range(m):
	print "Actual = %d   Predicted = %d  Manual = %d"%(y[i],a.predict(x[i].reshape(1,2)))

