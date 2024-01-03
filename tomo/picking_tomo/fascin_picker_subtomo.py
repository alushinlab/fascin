#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Trace filaments through a tomogram')
# IO parameters
parser.add_argument('--input_mrc_dir', type=str, help='Input file containing the tomogram in an MRC format')
parser.add_argument('--output_dir', type=str, help='Directory to store filament traces')
# Other parameters
parser.add_argument('--threshold', type=float, help='pixel threshold for binarization')
parser.add_argument('--pick_size', type=int, help='size of fascin in voxels cubed')
parser.add_argument('--angpix', type=float, help='Angstroms per pixel in input tomogram')
args = parser.parse_args()
print('')
if(args.input_mrc_dir == None or args.output_dir == None):
	print('Please enter an input_mrc, AND an output_dir')
	sys.exit('The preferred input style may be found with ./filament_tracer_tomo.py -h')

if(args.angpix == None or args.pick_size == None or args.threshold == None):
		sys.exit('Please enter all input parameters.')

input_mrc_dir = args.input_mrc_dir
outputDir = args.output_dir
threshold = args.threshold
pick_size = args.pick_size
angpix = args.angpix

if(input_mrc_dir[-1] != '/'): input_mrc_dir = input_mrc_dir + '/'
if(outputDir[-1] != '/'): outputDir = outputDir + '/'
print('Inputs accepted.')
print('Importing packages...')
################################################################################
# import of python packages
import numpy as np
import mrcfile
import json; import glob
import os; from tqdm import tqdm
from skimage import measure
import matplotlib.pyplot as plt
################################################################################
################################################################################
def get_centroids(file_name, THRESHOLD, MINIMUM_BLOB_SIZE):
	short_file_name = file_name.split('/')[-1]
	print('Loading tomogram ' + short_file_name + '...')
	with mrcfile.open(file_name) as mrc:
		real_data = mrc.data
	print('The dimensions of tomogram ' + short_file_name + ' are: ' + str(real_data.shape))
	real_data_binarized = real_data>THRESHOLD
	labels = measure.label(real_data_binarized)
	centroids = []
	for region in measure.regionprops(labels):
		if region.area > MINIMUM_BLOB_SIZE:
			centroids.append(region.centroid)
	
	return np.asarray(centroids)

def prune_hole(centroids, coord_X_px,coord_Y_px, radius_px):
	points_translated = centroids - np.asarray([0, coord_Y_px, coord_X_px])
	dist_sq = points_translated[:,1]**2 + points_translated[:,2]**2
	mask = dist_sq <= radius_px**2
	points_in_cylinder = centroids[mask]
	return points_in_cylinder

def remove_edges(centroids, file_name, edge_length):
	short_file_name = file_name.split('/')[-1]
	with mrcfile.open(file_name) as mrc:
		real_data = mrc.data
	tomo_shape = real_data.shape
	centroids_new = []
	for i in range(0, len(centroids)):
		if(centroids[i][0] > edge_length and centroids[i][1] > edge_length and centroids[i][2] > edge_length):
			if(centroids[i][0] < tomo_shape[0]-edge_length and centroids[i][1] < tomo_shape[1]-edge_length and centroids[i][2] < tomo_shape[2]-edge_length):
				centroids_new.append(centroids[i])
	
	centroids_new = np.asarray(centroids_new)
	print('The number of fascin blobs in tomogram ' + short_file_name + ' is: ' + str(len(centroids_new)))
	print('')
	return centroids_new

def starify(*args):
	return (''.join((('%.6f'%i).rjust(13))  if not isinstance(i,int) else ('%d'%i).rjust(13) for i in args) + ' \n')[1:]

def save_star_file(output_name, centroids):
	header = '# RELION; version 4.0\n\ndata_\n\nloop_ \n_rlnTomoName #1 \n_rlnCoordinateX #2 \n_rlnCoordinateY #3 \n_rlnCoordinateZ #4 \n'
	star_file = header
	tomo_name = output_name.split('/')[-1][:-5] + '   '
	for j in range(0, len(centroids)):
		star_file = star_file + (tomo_name + starify(centroids[j][2]*3.0,centroids[j][1]*3.0,centroids[j][0]*3.0))
	
	with open(output_name, "w") as text_file:
		text_file.write(star_file)

def save_extractions(output_name, centroids, full_tomo_name, edge_length):
	with mrcfile.open(full_tomo_name) as mrc:
		real_data = mrc.data
	for j in tqdm(range(0, len(centroids))):
		extraction = real_data[int(centroids[j][0])-edge_length:int(centroids[j][0])+edge_length, int(centroids[j][1])-edge_length:int(centroids[j][1])+edge_length, int(centroids[j][2])-edge_length:int(centroids[j][2])+edge_length]
		with mrcfile.new(output_name+'/%s_data.mrc'%str(j), overwrite=True) as mrc:
			mrc.set_data((extraction).astype('float32'))

def save_extractions(output_name, centroids, full_tomo_name, edge_length):
	with mrcfile.open(full_tomo_name) as mrc:
		real_data = mrc.data
	for j in tqdm(range(0, len(centroids))):
		extraction = real_data[int(centroids[j][0])-edge_length:int(centroids[j][0])+edge_length, int(centroids[j][1])-edge_length:int(centroids[j][1])+edge_length, int(centroids[j][2])-edge_length:int(centroids[j][2])+edge_length]
		mrcfile.write(output_name+'/%s_data.mrc'%str(j), (extraction).astype('float32'), overwrite=True, voxel_size=7.8)

import concurrent.futures
from tqdm import tqdm
def save_extractions(output_name, centroids, full_tomo_name, edge_length):
    with mrcfile.open(full_tomo_name) as mrc:
        real_data = mrc.data
    total_files = len(centroids)
    progress_bar = tqdm(total=total_files)
    def save_extraction(j):
        extraction = real_data[int(centroids[j][0])-edge_length:int(centroids[j][0])+edge_length, int(centroids[j][1])-edge_length:int(centroids[j][1])+edge_length, int(centroids[j][2])-edge_length:int(centroids[j][2])+edge_length]
        mrcfile.write(output_name+'/%s_data.mrc'%str(j), (extraction).astype('float32'), overwrite=True, voxel_size=7.8)
        progress_bar.update(1)

    with concurrent.futures.ThreadPoolExecutor(45) as executor:
        executor.map(save_extraction, range(len(centroids)))
    progress_bar.close()

################################################################################
# Center of carbon holes, manually measured, X,Y
cylinders = [[910,840],[500,910],[740,980],[-10,660],[580,1170],[570,1040],[555,765],
	     	 [610,980],[570,980],[550,920],[570,1000],[490,990],[530,1029],[590,1010],
			 [570,1030],[540,1000],[590,1080],[540,920],[550,980],[630,1020],[540,1000],
			 [610,1000],[560,1020],[920,300],[-200,790],[660,1040]]

################################################################################
# import data
print('All packages imported.')
print('This program will pick fascin centroids from a semantic segmentation map.')

# 
RADIUS_PX = 740
tot_picks = 0
mrc_file_names = sorted(glob.glob(input_mrc_dir+'/*channel1.mrc'))
mrc_file_names_denoised = sorted(glob.glob(input_mrc_dir+'../denoised/*.mrc'))
try:
	os.mkdir(outputDir+'Subtomograms/')
except FileExistsError:
	pass
for i in range(0, len(mrc_file_names)):
	centroids = get_centroids(mrc_file_names[i], threshold, pick_size)
	plt.scatter(centroids[:,2], centroids[:,1], s=1, alpha=0.5)
	centroids_prune_hole = prune_hole(centroids, cylinders[i][0],cylinders[i][1], RADIUS_PX)
	centroids_remove_edges = remove_edges(centroids_prune_hole, mrc_file_names[i], 32)
	plt.scatter(centroids_remove_edges[:,2], centroids_remove_edges[:,1], s=1, alpha=0.5)
	plt.axis('equal')
	plt.savefig(outputDir + 'plot_%s'%str(i).zfill(3) + '.png')
	plt.clf()
	print(centroids_remove_edges.shape)
	tot_picks = tot_picks + len(centroids_remove_edges)

	save_star_file(outputDir + mrc_file_names[i][:-20].split('/')[-1] + '.star', centroids_remove_edges)
	# save extractions
	try:
		os.mkdir(outputDir+'Subtomograms/'+mrc_file_names_denoised[i][:-11].split('/')[-1])
	except FileExistsError:
		pass
	save_extractions(outputDir + 'Subtomograms/'+mrc_file_names_denoised[i][:-11].split('/')[-1], centroids_remove_edges, mrc_file_names_denoised[i], 32)


print('The total number of picked particles is: ' + str(tot_picks))

print('The program has finished. Exiting...')


