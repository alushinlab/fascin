import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
####################################################################################################
all_data = np.loadtxt('_projections_along_eigenvectors_all_particles.txt', str)
e_vals = all_data[:,1].astype(float)


x1 = plt.hist(e_vals, bins=100)
plt.xlim(-25,25)
plt.show()


y_data = x1[0]
x_data = x1[1][:-1] - (x1[1][0] - x1[1][1])
plt.plot(x_data, y_data, 'o')

def Gauss(x, A, B):
	y = A*np.exp(-1*B*x**2)
	return y

def func(x, a, x0, sigma):
	return a*np.exp(-(x-x0)**2/(2*sigma**2))



parameters, covariance = curve_fit(func, x_data, y_data)


fit_A = parameters[0]
fit_B = parameters[1]
fit_C = parameters[2]

fit_y = func(x_data, fit_A, fit_B, fit_C)
#plt.plot(x_data, y_data, 'o', label='data')
plt.plot(x_data, fit_y, '-', label='fit')
plt.legend()
plt.show()

np.mean(e_vals[e_vals>5])
np.mean(e_vals[e_vals<-5])
np.mean(e_vals[np.logical_and(e_vals>-5, e_vals<5)])

# eigen-value of 0.18  == rotation of 20.06
# eigen-value of 7.79  == rotation of 30.67
# eigen-value of -7.63 == rotation of  9.07

# standard deviation of eigenvalues is 4.74
# average of eigenvalues is 0.18


plt.hist(e_vals, bins=100,color='lightblue')
plt.plot(x_data, fit_y, '-', label='fit', linewidth=3, color='blue')
plt.legend()
plt.show()



# Calculate the R-squared
# Calculate the Total Sum of Squares (SST)
sst = np.sum((y_data - np.mean(y_data)) ** 2)

# Calculate the Residual Sum of Squares (SSR)
ssr = np.sum((y_data - fit_y) ** 2)

# Calculate the R-squared
r_squared = 1 - (ssr / sst)

print(f"R-squared: {r_squared}")













