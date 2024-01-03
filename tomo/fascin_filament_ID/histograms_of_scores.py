#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import glob
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from scipy.stats import gamma
print('Imports finished. Beginning script...')
####################################################################################################
base_dir = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/filament_identification/histograms_PCscores/'
all_data = np.loadtxt(base_dir+'allEnergyScores.txt', str)
pc1_data = np.loadtxt(base_dir+'allEnergyScores_PC1.txt', str)

energyScores = all_data[:,1].astype(float)
pc1_scores = pc1_data[:,1].astype(float)

# Preliminary Plots
fig, ax = plt.subplots(1,2, figsize=(10,5))
x1 = ax[0].hist(energyScores, bins=100)
x2 = ax[1].hist(pc1_scores, bins=100)
plt.tight_layout()
plt.savefig(base_dir+'hist_allScores.png')
plt.show()
plt.clf()
####################################################################################################
# Fit gamma distribution
params = gamma.fit(energyScores)
fig, ax = plt.subplots(figsize=(10,5))
n, bins, patches = ax.hist(energyScores, bins=100, alpha=0.6, color='b')

# Plot fitted gamma distribution
x = np.linspace(min(energyScores), max(energyScores), 100)
y = gamma.pdf(x, params[0], loc=params[1], scale=params[2])
scale_factor = max(n) / max(y)
y_scaled = gamma.pdf(x, params[0], loc=params[1], scale=params[2])*scale_factor
print(np.max(y_scaled))

fig, ax = plt.subplots(1,2, figsize=(10,5))
x1 = ax[0].hist(energyScores, bins=100, color='lightblue')
ax[0].plot(x, y_scaled, '-', lw=3,color='black')
x2 = ax[1].hist(pc1_scores, bins=100)
plt.tight_layout()
#plt.savefig(base_dir+'hist_allScores.png')
plt.clf()


####################################################################################################
# Fit gamma distribution and color!
colors = [(1, 1, 1), (1, 0, 0)]  # This corresponds to [white, red]
n_bins = 1000  # Discretizes the interpolation into bins
import matplotlib.colors as mcolors
cmap_name = 'custom_white_to_red'
cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
norm = plt.Normalize(vmin=1.5, vmax=3.0)# 2 and 2.4 for each fascin


params = gamma.fit(energyScores)
counts, bin_edges = np.histogram(energyScores, bins=100)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_colors = cmap(norm(bin_centers))

fig, ax = plt.subplots(1,2, figsize=(10,5))
x1 = ax[0].bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), color=bin_colors, align='center',edgecolor='black',linewidth=0.1)
ax[0].plot(x, y_scaled, '-', lw=3,color='black')
x2 = ax[1].hist(pc1_scores, bins=100, color='gray')
ax[0].set_ylim(0,4000)
ax[1].set_ylim(0,4000)
plt.tight_layout()
#plt.savefig(base_dir+'hist_allScores.png')
#plt.savefig(base_dir+'hist_allScores.svg')
plt.show()

# Compute R-squared
# Calculate predicted frequencies using the fitted gamma distribution parameters
predicted_counts = gamma.pdf(bin_centers, params[0], loc=params[1], scale=params[2]) * (bin_edges[1] - bin_edges[0]) * len(energyScores)
# Calculate SST (Total Sum of Squares)
sst = np.sum((counts - np.mean(counts))**2)
# Calculate SSR (Residual Sum of Squares)
ssr = np.sum((counts - predicted_counts)**2)
# Calculate R-squared
r_squared = 1 - ssr / sst
print(f'R-squared: {r_squared}')
sys.exit()











####################################################################################################
# Fit gaussian to PC1
#pc1_scores = pc1_scores / np.max(pc1_scores)
x2 = plt.hist(pc1_scores, bins=100)
plt.clf()  # Clear the plot if you don't want to show this histogram

# Extract the data we need for GMM fitting
y_data = x2[0]
bin_edges = x2[1]
x_data = 0.5 * (bin_edges[1:] + bin_edges[:-1])
X = np.repeat(x_data,y_data.astype(int))

# Create a Gaussian Mixture Model with two components
gmm = GaussianMixture(n_components=1, random_state=0)

# We will fit the GMM such that each sample's weight is its corresponding bin count
gmm.fit(X.reshape(-1,1))

# Generating a smooth line for each component and the combined GMM
x_values = np.linspace(x_data.min(), x_data.max(), 100).reshape(-1, 1)
gmm_scores = gmm.score_samples(x_values)
gmm_y = np.exp(gmm_scores) * y_data.sum() * (x_values[1] - x_values[0])  # Scale it to the histogram

# Plot the original histogram
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(energyScores, bins=100)
ax[1].hist(pc1_scores, bins=100, color='lightblue', label='Histogram')

# Plot each Gaussian component
for mean, covar, weight in zip(gmm.means_, gmm.covariances_, gmm.weights_):
    component_curve = weight * np.exp(-0.5 * ((x_values - mean)**2) / covar) / np.sqrt(2 * np.pi * covar)
    ax[1].plot(x_values, component_curve.flatten() * y_data.sum() * (x_values[1] - x_values[0]),
                label=f'Component {mean[0]:.2f}', linewidth=2)

# Plot the combined GMM
ax[1].plot(x_values, gmm_y, label='GMM Fit', color='red', linewidth=2)

# Add labels and legends
ax[1].set_xlabel('PC1 Scores')
ax[1].set_ylabel('Frequency')
ax[1].legend()

plt.tight_layout()
plt.savefig(base_dir+'hist_allScores.png')

####################################################################################################
sys.exit()


















x2 = plt.hist(pc1_scores, bins=100)
plt.clf()
y_data = x2[0]
x_data = x2[1][:-1] - (x2[1][0] - x2[1][1])


def Gauss(x, A, B):
	y = A*np.exp(-1*B*x**2)
	return y

def func(x, a, x0, sigma):
	return a*np.exp(-(x-x0)**2/(2*sigma**2))

initial_guess = [max(y_data), 0,5000]
parameters, covariance = curve_fit(func, x_data, y_data, p0=initial_guess)

fit_A = parameters[0]
fit_B = parameters[1]
fit_C = parameters[2]

fit_y = func(x_data, fit_A, fit_B, fit_C)
# Done fitting Gaussian to PC1

fig, ax = plt.subplots(1,2, figsize=(10,5))
x1 = ax[0].hist(energyScores, bins=100)
x2 = ax[1].hist(pc1_scores, bins=100, color='lightblue')
plt.plot(x_data, fit_y, '-', label='fit', linewidth=3, color='blue')
plt.tight_layout()
plt.savefig(base_dir+'hist_allScores.png')
plt.show()