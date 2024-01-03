#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
################################################################################
# import of python packages
import numpy as np
import warnings
import mrcfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy.ndimage import filters
################################################################################
# save coordinates as bild file
def generate_bild(tomo_coords_and_colors):
	base_name = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/'
	o= base_name + tomo_coords_and_colors[0][0] +'.bild'
	out=open(o, 'w')
	out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
	for j in range(0, len(tomo_coords_and_colors)):
			#write out marker entries for each residue pair
			out.write('.color %.5f %.5f %.5f\n'%(tomo_coords_and_colors[j][6].astype(float), tomo_coords_and_colors[j][7].astype(float), tomo_coords_and_colors[j][8].astype(float)))
			out.write(".sphere %.5f %.5f %.5f %.5f \n"%(tomo_coords_and_colors[j][2].astype(float), tomo_coords_and_colors[j][3].astype(float), tomo_coords_and_colors[j][4].astype(float), 6))
	#write final line of xml file, is constant	
	out.close()	

def mahalanobis_dist_12dim(points, eigenvectors, lambdas):
	LAMBDA = np.diag(lambdas)
	SIGMA = eigenvectors.T @ LAMBDA @ eigenvectors / eigenvectors.shape[0]
	# Plot the correlation matrix as a heatmap
	plt.imshow(SIGMA, cmap='coolwarm', interpolation='nearest')
	plt.colorbar()
	plt.title('Covariance Matrix Heatmap')
	plt.xlabel('Dimensions')
	plt.ylabel('Dimensions')
	plt.savefig('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/corrMatrix_12dim.png', format='png')
	plt.clf()
	points = points[:,1:].astype(float)
	mahalanobis_distances = np.zeros(points.shape[0])
	for i in tqdm(range(0, len(points))):
		mahalanobis_distances[i] = np.sqrt(points[i] @ np.linalg.inv(SIGMA) @ points[i].T)
	
	plt.hist(mahalanobis_distances, bins=100)
	plt.savefig('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/scores_12dim.png', format='png')
	return np.expand_dims(mahalanobis_distances,axis=-1)

def mahalanobis_dist_6dim(points):
	from scipy.stats import gamma
	six_dim_subspace_pts = points[:,[1,2,3,4,5,8]].astype(float)
	euclidean_average = np.mean(six_dim_subspace_pts, axis=0)
	mean_centered_points = six_dim_subspace_pts - euclidean_average
	SIGMA = np.cov(mean_centered_points, rowvar=False)
	mahalanobis_distances = np.zeros(six_dim_subspace_pts.shape[0])
	for i in tqdm(range(0, len(mean_centered_points))):
		mahalanobis_distances[i] = np.sqrt(mean_centered_points[i] @ np.linalg.inv(SIGMA) @ mean_centered_points[i].T)
	
	fig, ax1 = plt.subplots()
	ax1.hist(mahalanobis_distances, bins=np.linspace(0, 10, 100), density=True, alpha=0.6, color='g', label='Empirical Data')
	ax1.set_xlabel('Mahalanobis Distance')
	ax1.set_ylabel('Probability Density (Histogram)', color='g')
	ax1.set_ylim(0, 0.5)
	ax1.set_xlim(0, 8)
	# Fit a gamma distribution to the data
	shape, loc, scale = gamma.fit(mahalanobis_distances, f0=6, floc=0)
	gamma_pdf = gamma.pdf(np.linspace(0, 10, 100), shape, loc=loc, scale=scale)
	# Plot the gamma PDF
	ax2 = ax1.twinx()
	ax2.plot(np.linspace(0, 10, 100), gamma_pdf, 'r-', lw=2, label='Gamma Distribution')
	ax2.set_ylabel('Probability Density (Gamma PDF)', color='r')
	# Combine the legends from both axes
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc='upper right')
	ax2.set_ylim(0, 0.5)
	ax2.set_xlim(0, 8)
	plt.savefig('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/scores.png', format='png')

	potential_values = -np.log(gamma.pdf(np.linspace(0, 10, 100), shape, loc=loc, scale=scale))
	
	fig, ax3 = plt.subplots()
	ax3.plot(np.linspace(0, 10, 100), potential_values, 'b-', lw=2, label='Potential Values')
	ax3.set_xlabel('Mahalanobis Distance')
	ax3.set_ylabel('Potential Values', color='b')
	ax3.legend(loc='upper left')
	ax3.set_ylim(0, np.max(potential_values[1:]) * 1.2)  # Adjust ylim for better visualization
	ax3.set_xlim(0, 8)
	plt.savefig('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/potentials.png', format='png')

	return np.expand_dims(mahalanobis_distances, axis=-1)

def prune_hole(radius_px, coord_X_px,coord_Y_px, shape):
    mask = np.ones(shape, dtype=np.float32)
    Z, Y, X = shape
    # Create coordinate grids for Z, Y, and X axes
    z_grid, y_grid, x_grid = np.ogrid[:Z, :Y, :X]
    # Calculate distances from the center of the cylinder
    distances = np.sqrt((x_grid - coord_X_px) ** 2 + (y_grid - coord_Y_px) ** 2)[0]
    # Set values outside the cylinder to zero
    mask[:,distances > radius_px] = 0
    return mask

def prune(skel_pruned, i):
	if(i==0):
		return skel_pruned
	else:
		cube = np.ones((3,3,3))
		cube[1,1,1] = 28
		E_img = filters.convolve(skel_pruned,cube)
		skel_pruned[E_img ==29] = 0
		prune(skel_pruned, i-1)
	return skel_pruned

def skeletonize_and_prune_nubs(skel, intersect_removal_box_size, prune_length, skel_threshold):
	skel_pruned = prune(skel.copy(), prune_length)
	# Follow up pruning with another round of skeletonize to remove weird triple pts
	skel_pruned = skeletonize_3d(skel_pruned.astype('float32')>skel_threshold-0.05)
	skel_pruned[skel_pruned>0] = 1
	
	# Now, remove three-way and four-way intersections
	cube = np.ones((3,3,3))
	cube[1,1,1] = 28	
	triple_pts = np.asarray(np.argwhere(cube==29))
	for i in range(0, len(triple_pts)):
		x = triple_pts[i][0]
		y = triple_pts[i][1]
		skel_pruned[x-intersect_removal_box_size:x+intersect_removal_box_size,y-intersect_removal_box_size:y+intersect_removal_box_size] = 0
	return skel_pruned

def remove_small_blobs(map, THRESHOLD, MINIMUM_BLOB_SIZE):
	map_binarized = map > THRESHOLD
	skeleton = morphology.skeletonize_3d(map_binarized)
	pruned_skeleton = prune(skeleton, 30)
	labels = measure.label(pruned_skeleton)
	regions = []
	segmented_volume = np.zeros_like(labels)
	for region in measure.regionprops(labels):
		if region.area > MINIMUM_BLOB_SIZE:
			region_id = region.label
			segmented_volume[labels == region_id] = region_id
	#		regions.append(region)
	
	return segmented_volume, 0#np.asarray(regions)


################################################################################
################################################################################
################################################################################
# import data
eigenvalue_file_name = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/final_bin1_MBR_projections_along_eigenvectors_all_particles.txt'
pickedCoord_file_name = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/PseudoSubtomo/job044/particles.star'
eigenvectors_file_name = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/final_bin1_MBR_eigenvectors.dat'
eigenvalues = np.loadtxt(eigenvalue_file_name,dtype=str)
picked_coords = np.loadtxt(pickedCoord_file_name,dtype=str,skiprows=45)
eigenvectors = np.loadtxt(eigenvectors_file_name,dtype=float, skiprows=1)
actual_lambdas = np.asarray([0.2588,0.1272,0.1080,0.1021,0.0853,0.0661,0.0655,0.0643,0.0435,0.0403,0.0238,0.015])

# Take subspace of eigenvalues
energy_scores = mahalanobis_dist_6dim(eigenvalues)
combo1 = np.concatenate((np.expand_dims(eigenvalues[:,0], axis=-1),energy_scores,picked_coords), axis=-1)
energy_scores = eigenvalues[:,[1]].astype(float)
combo = np.concatenate((np.expand_dims(eigenvalues[:,0], axis=-1),energy_scores,picked_coords), axis=-1)
np.savetxt('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/filament_identification/allEnergyScores.txt', combo1, fmt=' '.join(['%s']*23))
np.savetxt('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/filament_identification/allEnergyScores_PC1.txt', combo, fmt=' '.join(['%s']*23))

# verify combined correctly
for i in range(0, len(combo)):
    if(combo[i][0] != combo[i][12]):
        print('Entry ' + combo[i][0] + ' does not match entry ' + combo[i][12])

# Chunk into separate tomograms
unique_entries = np.unique(combo[:,2])
chunked_picks = []
for entry in tqdm(unique_entries):
	indices = np.char.find(combo[:,2], entry) >= 0
	chunked_array = combo[indices]
	chunked_picks.append(chunked_array)

# Retain only relevant columns
for i in range(0, len(chunked_picks)):
	chunked_picks[i] = chunked_picks[i][:,[2,6,3,4,5,1]]

print(chunked_picks[0][0])

# Create a normalization instance to map data to the range [0, 1]
print((np.percentile(energy_scores, 70), np.percentile(energy_scores, 99)))
norm = plt.Normalize(vmin=np.percentile(energy_scores, 40), vmax=np.percentile(energy_scores, 90))#vmax=np.max(energy_scores)*0.9)  # Adjust the range as needed

# Create a colormap instance
cmap = plt.cm.get_cmap('Reds')
print('40th Percentile: ' + str(np.percentile(energy_scores, 40)))
print('90th Percentile: ' + str(np.percentile(energy_scores, 90)))
coords_and_colors = []
for i in range(0, len(chunked_picks)):
	# Map your data to the colormap using the normalization
	#mapped_colors = cmap(norm(np.abs(chunked_picks[i][:,-1].astype(float))))
	mapped_colors = cmap(norm(chunked_picks[i][:,-1].astype(float)))
	coords_and_colors.append(np.concatenate((chunked_picks[i], mapped_colors), axis=-1))

print(coords_and_colors[0][0])
for i in range(0, len(coords_and_colors)):
	generate_bild(coords_and_colors[i])

'''
################################################################################
# Now do filament tracing
################################################################################
# Center of carbon holes, manually measured, X,Y
cylinders = [[910,840],[500,910],[740,980],[-10,660],[580,1170],[570,1040],[555,765],
	     	 [610,980],[570,980],[550,920],[570,1000],[490,990],[530,1029],[590,1010],
			 [570,1030],[540,1000],[590,1080],[540,920],[550,980],[630,1020],[540,1000],
			 [610,1000],[560,1020],[920,300],[-200,790],[660,1040]]

RADIUS_PX = 740
################################################################################
semMap_file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/semSeg_actin/*0.mrc'))
semMap_file_names = semMap_file_names[:2]
semSeg_maps = []
print('Loading semSeg maps...')
for i in tqdm(range(0, len(semMap_file_names))):
	with mrcfile.open(semMap_file_names[i]) as mrc:
		semSeg_maps.append(mrc.data)

print(len(cylinders))
print(len(semSeg_maps))
# mask the semSeg maps
print('Masking the semantic segmentation maps with cylindrical masks')
for i in tqdm(range(0, len(semSeg_maps))):
	mask = prune_hole(RADIUS_PX, cylinders[i][0], cylinders[i][1], semSeg_maps[i].shape)
	maskedSemSegMap = np.multiply(semSeg_maps[i], mask)
	removedFloaters, regions = remove_small_blobs(maskedSemSegMap, 0.98, 1000)
	
	mrcfile.write(semMap_file_names[i][:-4]+'_masked.mrc', (removedFloaters).astype('float32'), overwrite=True, voxel_size=7.8)




print('hi')
print('hi')
'''