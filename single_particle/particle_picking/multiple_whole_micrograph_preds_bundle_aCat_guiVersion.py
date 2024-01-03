#!/home/greg/.conda/envs/matt_TF/bin/python
################################################################################
print('Loading python packages...')
################################################################################
# import of python packages
import numpy as np
import matplotlib.pyplot as plt
import keras
import mrcfile
import random
from tqdm import tqdm
from keras import layers
from keras.models import Model
import tensorflow as tf
#from EMAN2 import *
from scipy import interpolate; from scipy.ndimage import filters
from skimage.morphology import skeletonize_3d; import scipy
import keras.backend as K
import glob
import os
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Picking bundle particles.')
parser.add_argument('--semseg_net_dir', type=str, help='Path to trained semseg network')
parser.add_argument('--binned_micro_dir', type=str, help='Path to binned micrographs')
parser.add_argument('--hist_match_dir', type=str, help='Path to histogram matching image')
parser.add_argument('--increment', type=int, help='Determines overlap of unit boxes')
parser.add_argument('--box_size', type=int, help='Size of unit box')
parser.add_argument('--fudge_fac', type=float, help = 'Fudge factor')
parser.add_argument('--sq_size', type=int, help='output directory to store noisy 2D projections')
parser.add_argument('--threshold', type=float, help='Box size of projected image')
parser.add_argument('--gpu_idx', type=int, help='Number of GPUs available to run picking')
parser.add_argument('--occ_high', type=float, help='output directory to store noiseless 2D projections')
parser.add_argument('--occ_low', type=float, help='output directory to store semantic segmentation targets')
parser.add_argument('--radius', type=float, help='Determines spacing of particle picks')
parser.add_argument('--innerbox', type=int, help='total number of parallel threads to launch')
parser.add_argument('--occ_smallBox', type=float, help='Box size of projected image')
parser.add_argument('--output_dir', type=str, help='Path of folder where picks are placed')

args = parser.parse_args()
SEMSEG_DIR_PATH = args.semseg_net_dir              #'./semSeg_50ktrain_catCrossEnt.h5'
BINNED_MICRO_PATH = args.binned_micro_dir          #'../Micrographs_bin4/*_bin4.mrc'
HIST_MATCH_PATH = args.hist_match_dir              #'./actin_rotated%05d.mrc'
INCREMENT = int(args.increment)                    #48
BOX_SIZE = int(args.box_size)                      #96
FUDGE_FAC = float(args.fudge_fac)                    #0.9
SQ_SIZE = int(args.sq_size)                        #96
HALF_SQ_SIZE = int(SQ_SIZE / 2)
THRESHOLD = float(args.threshold)                    #192
GPU_IDX = int(args.gpu_idx)
OCC_HIGH = float(args.occ_high)                    #0.5
OCC_LOW = float(args.occ_low)                      #0.10 # 0.15
RADIUS = int(args.radius)                          #36
INNER_BOX = int(args.innerbox)                    #12
INNER_BOX_HALF = int(INNER_BOX / 2)
OCC_SMALLBOX = float(args.occ_smallBox)            #0.80
OUTPUT_DIR_PATH = args.output_dir                  #should be like 'PickParticles/job005'


if(OUTPUT_DIR_PATH[-1] != '/'): OUTPUT_DIR_PATH = OUTPUT_DIR_PATH + '/'

print('All inputs have been entered properly...')
print('Setting up output directories...')
# make directories
if(not os.path.isdir(OUTPUT_DIR_PATH+'pngs/')): os.mkdir(OUTPUT_DIR_PATH+'pngs/')
if(not os.path.isdir(OUTPUT_DIR_PATH+'pngs_semSeg/')): os.mkdir(OUTPUT_DIR_PATH+'pngs_semSeg/')
if(not os.path.isdir(OUTPUT_DIR_PATH+'starFiles/')): os.mkdir(OUTPUT_DIR_PATH+'starFiles/')

print('The program will now run.')
################################################################################
print('Python packages loaded. Setting CUDA environment...')
#GPU_IDX = 2
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_IDX)
################################################################################
################# Functions for opterating on whole micrographs ################
def slice_up_micrograph(real_data, increment, box_size,hist_matcher):#, box_size):
	extractions = []
	for i in range(0, real_data.shape[0]-box_size, increment):
		for j in range(0, real_data.shape[1]-box_size, increment):
			extraction = -1.0*real_data[i:i+box_size,j:j+box_size]
			extraction = hist_match(extraction, hist_matcher)
			extractions.append(extraction)
	extractions = np.moveaxis(np.asarray(extractions), 0,-1)
	extractions = (extractions - np.mean(extractions, axis=(0,1))) / np.std(extractions, axis=(0,1))
	return np.moveaxis(extractions, -1, 0)

def stitch_back_seg(shape, preds, increment, box_size):
	stitch_back = np.zeros((shape))
	cntr=0
	for i in range(0, stitch_back.shape[0]-box_size, increment):
		for j in range(0, stitch_back.shape[1]-box_size, increment):
			stitch_back[i:i+box_size, j:j+box_size] = np.max(np.stack((preds[cntr], stitch_back[i:i+box_size, j:j+box_size])),axis=0)
			cntr=cntr+1
	return stitch_back

def hist_match(source, template):
	oldshape = source.shape
	source = source.ravel()
	template = template.ravel()
	# get the set of unique pixel values and their corresponding indices and counts
	s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
	t_values, t_counts = np.unique(template, return_counts=True)
	
	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]
	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]
	
	# interpolate linearly to find the pixel values in the template image
	# that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
	return interp_t_values[bin_idx].reshape(oldshape)
'''
def implement_NMS(semMap):
	################################################################################
	# Four parameters
	
	sq_size = 96
	half_box_size = int(box_size / 2)
	occ_high = 0.5
	occ_low = 0.10 # 0.15
	radius = 36
	inner_box = 12
	inner_box_half = int(inner_box / 2)
	occ_smallBox = 0.80
	
	################################################################################
	picks = np.argwhere(semMap==1)
	picks_noborder = [] # make more efficient with np.where
	for x in range(0,picks.shape[0]):
		if picks[x][0] >= HALF_SQ_SIZE and picks[x][1] >= HALF_SQ_SIZE and picks[x][0] < semMap.shape[0]-HALF_SQ_SIZE and picks[x][1] < semMap.shape[1]-HALF_SQ_SIZE:
			picks_noborder.append(picks[x])
	
	picks_noborder = np.asarray(picks_noborder)
	
	picks_occupancy = []
	box_area = float(SQ_SIZE**2)
	for x in (range(0,len(picks_noborder))):
		box = semMap[picks_noborder[x][0]-HALF_SQ_SIZE:picks_noborder[x][0]+HALF_SQ_SIZE,picks_noborder[x][1]-HALF_SQ_SIZE:picks_noborder[x][1]+HALF_SQ_SIZE]
		occupancy = np.sum(box) / box_area
		if occupancy > OCC_LOW and occupancy < OCC_HIGH:
			picks_occupancy.append(picks_noborder[x])
	
	picks_occupancy = np.asarray(picks_occupancy)
	################################################################################
	picks_occupancy_2 = []
	smallBox_area = float(INNER_BOX**2)
	for x in (range(0,len(picks_occupancy))):
		box = semMap[picks_occupancy[x][0]-INNER_BOX_HALF:picks_occupancy[x][0]+INNER_BOX_HALF,picks_occupancy[x][1]-INNER_BOX_HALF:picks_occupancy[x][1]+INNER_BOX_HALF]
		occupancy = np.sum(box) / smallBox_area
		if occupancy > OCC_SMALLBOX:
			picks_occupancy_2.append(picks_occupancy[x])
	
	picks_occupancy_2 = np.asarray(picks_occupancy_2)
	picks_occupancy_2 = picks_occupancy_2
	
	pick_com = []
	for x in (range(0, len(picks_occupancy_2))):
		box = semMap[picks_occupancy_2[x][0]-HALF_SQ_SIZE:picks_occupancy_2[x][0]+HALF_SQ_SIZE,picks_occupancy_2[x][1]-HALF_SQ_SIZE:picks_occupancy_2[x][1]+HALF_SQ_SIZE]
		com = center_of_mass(box)
		if(np.abs(com[0]-HALF_SQ_SIZE) < 3 or np.abs(com[1]-HALF_SQ_SIZE) < 3):
			pick_com.append(picks_occupancy_2[x])
	
	pick_com = np.asarray(pick_com)
	semMap_filtered = semMap.copy()
	for i in range(0, len(pick_com)):
		semMap_filtered[pick_com[i][0],pick_com[i][1]] = 2
	
	semMap_filtered = semMap_filtered-1
	semMap_filtered[semMap_filtered == -1] = 0
	#semMap_skel = skeletonize_3d(semMap_filtered)
	#picks_skel = np.argwhere(semMap_skel==255)
	picks_skel = np.argwhere(semMap_filtered==1)
	# implement NMS algorithm
	nms = picks_skel.copy()[::128]
	project_dist_matrix=1
	if(len(nms) == 0):
		return []
	while(np.sum(project_dist_matrix) != 0):
		sq_len = len(nms)
		coord1 = np.repeat(np.expand_dims(nms, axis=0),  len(nms), axis=0).reshape(sq_len,sq_len,2)
		coord2 = np.repeat(np.expand_dims(nms, axis=-1), len(nms), axis=0).reshape(sq_len,sq_len,2)
		dist_matrix = np.linalg.norm(coord1 - coord2, axis=-1)
		neighbor_matrix = dist_matrix < RADIUS # cutoff dist for being called a neighbor
		dist_matrix_binarized = np.multiply(-1*np.eye(len(dist_matrix))+1,neighbor_matrix)
		project_dist_matrix = np.sum(dist_matrix_binarized,axis=0)
		val_to_pop = np.argmax(project_dist_matrix)
		nms = np.delete(nms, val_to_pop, axis=0)
	
	return nms
'''
def implement_NMS(semMap):
	################################################################################
	# Four parameters
	'''
	sq_size = 96
	half_box_size = int(box_size / 2)
	occ_high = 0.5
	occ_low = 0.10 # 0.15
	radius = 36
	inner_box = 12
	inner_box_half = int(inner_box / 2)
	occ_smallBox = 0.80
	'''
	################################################################################
	picks = np.argwhere(semMap==1)
	picks_noborder = [] # make more efficient with np.where
	for x in range(0,picks.shape[0]):
		if picks[x][0] >= HALF_SQ_SIZE and picks[x][1] >= HALF_SQ_SIZE and picks[x][0] < semMap.shape[0]-HALF_SQ_SIZE and picks[x][1] < semMap.shape[1]-HALF_SQ_SIZE:
			picks_noborder.append(picks[x])
	
	picks_noborder = np.asarray(picks_noborder)
	
	picks_occupancy = []
	box_area = float(SQ_SIZE**2)
	for x in (range(0,len(picks_noborder))):
		box = semMap[picks_noborder[x][0]-HALF_SQ_SIZE:picks_noborder[x][0]+HALF_SQ_SIZE,picks_noborder[x][1]-HALF_SQ_SIZE:picks_noborder[x][1]+HALF_SQ_SIZE]
		occupancy = np.sum(box) / box_area
		if occupancy >= OCC_LOW and occupancy <= OCC_HIGH:
			picks_occupancy.append(picks_noborder[x])
	
	picks_occupancy = np.asarray(picks_occupancy)
	################################################################################
	'''
	picks_occupancy_2 = []
	smallBox_area = float(INNER_BOX**2)
	for x in (range(0,len(picks_occupancy))):
		box = semMap[picks_occupancy[x][0]-INNER_BOX_HALF:picks_occupancy[x][0]+INNER_BOX_HALF,picks_occupancy[x][1]-INNER_BOX_HALF:picks_occupancy[x][1]+INNER_BOX_HALF]
		occupancy = np.sum(box) / smallBox_area
		if occupancy > OCC_SMALLBOX:
			picks_occupancy_2.append(picks_occupancy[x])
	
	picks_occupancy_2 = np.asarray(picks_occupancy_2)
	picks_occupancy_2 = picks_occupancy_2
	
	pick_com = []
	for x in (range(0, len(picks_occupancy_2))):
		box = semMap[picks_occupancy_2[x][0]-HALF_SQ_SIZE:picks_occupancy_2[x][0]+HALF_SQ_SIZE,picks_occupancy_2[x][1]-HALF_SQ_SIZE:picks_occupancy_2[x][1]+HALF_SQ_SIZE]
		com = center_of_mass(box)
		if(np.abs(com[0]-HALF_SQ_SIZE) < 3 or np.abs(com[1]-HALF_SQ_SIZE) < 3):
			pick_com.append(picks_occupancy_2[x])
	'''
	pick_com = np.asarray(picks_occupancy)
	semMap_filtered = semMap.copy()
	for i in range(0, len(pick_com)):
		semMap_filtered[pick_com[i][0],pick_com[i][1]] = 2
	
	semMap_filtered = semMap_filtered-1
	semMap_filtered[semMap_filtered == -1] = 0
	#semMap_skel = skeletonize_3d(semMap_filtered)
	#picks_skel = np.argwhere(semMap_skel==255)
	picks_skel = np.argwhere(semMap_filtered==1)
	# implement NMS algorithm
	nms = picks_skel.copy()[::64]
	project_dist_matrix=1
	if(len(nms) == 0):
		return []
	
	sq_len = len(nms)
	coord1 = np.repeat(np.expand_dims(nms, axis=0),  len(nms), axis=0).reshape(sq_len,sq_len,2)
	coord2 = np.repeat(np.expand_dims(nms, axis=-1), len(nms), axis=0).reshape(sq_len,sq_len,2)
	dist_matrix = np.linalg.norm(coord1 - coord2, axis=-1)
	
	not_finished = True
	i = 0
	while(not_finished):
		neighbor_matrix = dist_matrix < RADIUS # cutoff dist for being called a neighbor
		dist_matrix_binarized = np.multiply(-1*np.eye(len(dist_matrix))+1,neighbor_matrix)
		too_close = np.argwhere(dist_matrix_binarized[i] == 1)
		# remove all elements of too_close from dist_matrix
		dist_matrix = np.delete(np.delete(dist_matrix, too_close, axis=0), too_close, axis=1)
		nms = np.delete(nms, too_close, axis=0)
		i = i + 1
		if(i >= len(dist_matrix)):
			not_finished=False
			break
	
	return nms




def starify(*args):
	return (''.join((('%.6f'%i).rjust(13))  if not isinstance(i,int) else ('%d'%i).rjust(13) for i in args) + ' \n')[1:]

################################################################################
# load trained Fully Convolutional Network for semantic segmentation
################################################################################
print('Loading neural network models'); print('')
#model_path = './semSeg_50ktrain_catCrossEnt.h5'
FCN = keras.models.load_model(SEMSEG_DIR_PATH)

print('Network loaded')
# Load one test image for histogram matching
#hist_match_dir = './'
with mrcfile.open(HIST_MATCH_PATH) as mrc:
	hist_matcher = mrc.data

################################################################################
def run_pick_on_micrograph(file_name):
	big_micrograph_name = file_name
	with mrcfile.open(big_micrograph_name) as mrc:
		real_data = mrc.data
	
	################################################################################
	# Divide up the whole micrograph to feed to the FCN for semantic segmentation
	#increment = 48
	extractions = slice_up_micrograph(real_data, INCREMENT, BOX_SIZE, hist_matcher)
	preds = FCN.predict(np.expand_dims(FUDGE_FAC*extractions, axis=-1))
	stitch_back = stitch_back_seg(real_data.shape, preds[:,:,:,2], INCREMENT, BOX_SIZE)
	with mrcfile.new(OUTPUT_DIR_PATH+'pngs_semSeg/'+big_micrograph_name[:-4].split('/')[-1]+'.mrc', overwrite=True) as mrc:
		mrc.set_data((stitch_back>THRESHOLD).astype('float32'))
	
	binarized_stitch_back = (stitch_back > THRESHOLD).astype('float32')
	picks = implement_NMS(binarized_stitch_back)
	
	real_data = gaussian_filter(real_data, 1)
	middle_box = real_data[int(real_data.shape[0]*0.25):int(real_data.shape[0]*0.75), int(real_data.shape[1]*0.25):int(real_data.shape[1]*0.75)]
	real_data = real_data - np.min(middle_box)
	_=plt.imshow(real_data, origin='lower', cmap=plt.cm.gray, vmin=0, vmax=0.95*(np.max(middle_box)-np.min(middle_box)))
	if(picks != []): _=plt.scatter(picks[:,1], picks[:,0], s=18, c='springgreen')
	_=plt.tight_layout()
	_=plt.savefig(OUTPUT_DIR_PATH+'pngs/'+big_micrograph_name[:-4].split('/')[-1]+'.png', dpi=600)
	#plt.show()
	_=plt.clf()
	
	################################################################################
	# Prepare star file
	header = '# RELION; version 3.0\n\ndata_\n\nloop_ \n_rlnCoordinateX #1 \n_rlnCoordinateY #2 \n_rlnAngleTiltPrior #3 \n'
	star_file = header
	for j in range(0, len(picks)):
		star_file = star_file + starify(picks[j][1]*4.0,picks[j][0]*4.0,90.0)
	
	star_path = OUTPUT_DIR_PATH+'starFiles/'+(big_micrograph_name[:-4].split('/')[-1]).split('_bin4')[0]
	with open(star_path+'.star', "w") as text_file:
		text_file.write(star_file)

# Load real micrographs
if('.txt' in BINNED_MICRO_PATH):
	with open(BINNED_MICRO_PATH) as file:
		file_names = file.readlines()
		file_names = [line.rstrip() for line in file_names]

else:
	file_names = sorted(glob.glob(BINNED_MICRO_PATH))

pngs = sorted(glob.glob('./particle_picking/job045/pngs/*.png'))
pngs_trimmed = []
for i in range(0, len(pngs)):
	pngs_trimmed.append(pngs[i][31:-4])

print(pngs_trimmed)
sys.exit()


skipped_file_names = []
for i in range(0, len(micrographs_bin4)):
	if(micrographs_bin4[i][20:-4] not in set(pngs_trimmed)):
		skipped_file_names.append(micrographs_bin4[i])



'''
import random 
random.seed(4)
file_names = random.sample(file_names, 100)
'''

img_num = len(skipped_file_names)#len(sorted(glob.glob('../Micrographs_bin4/*_bin4.mrc')))
img_start = (img_num / 4)*GPU_IDX
img_end = (img_num / 4)*(GPU_IDX+1)
file_names = skipped_file_names[img_start:img_end]
for i in tqdm(range(0, len(file_names)), file=sys.stdout):
	run_pick_on_micrograph(file_names[i])


print('Exiting...')







