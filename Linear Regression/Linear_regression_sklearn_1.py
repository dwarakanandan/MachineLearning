import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr

data = np.loadtxt("ex1data1.txt",delimiter = ',')
m = data[:,0].size
x = data[:,0].reshape(m,1)
y = data[:,1]
a = lr(fit_intercept=True)
a.fit(x,y)
print a.coef_
print a.intercept_
print a.score(x,y)
plt.scatter(x,y)
plt.plot(x,a.predict(x))
plt.show()

