#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
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
from scipy import interpolate; from scipy.ndimage import filters
from skimage.morphology import skeletonize_3d; import scipy
import keras.backend as K
import glob
import os
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter
import threading
import argparse; import sys
from tomo_picker_functions import *
#from tomo_picker_functions_testingScaleFactor import *
try:
	import thread
except ImportError:
	import _thread as thread

################################################################################
# parse passed arguments
parser = argparse.ArgumentParser('Picking bundle particles.')
parser.add_argument('--dae_net_dir', type=str, help='Path to trained semseg network')
parser.add_argument('--semseg_net_dir', type=str, help='Path to trained semseg network')
parser.add_argument('--binned_micro_dir', type=str, help='Path to binned micrographs')
parser.add_argument('--hist_match_dir', type=str, help='Path to histogram matching image')
parser.add_argument('--increment', type=int, help='Determines overlap of unit boxes')
parser.add_argument('--box_size', type=int, help='Size of unit box')
parser.add_argument('--fudge_fac', type=float, help = 'Fudge factor')
parser.add_argument('--threshold', type=float, help='Box size of projected image')
parser.add_argument('--gpu_idx', type=int, help='GPU index to launch this process')
parser.add_argument('--tot_gpus', type=int, help='Total number of GPUs available to run picking across all nodes')
#parser.add_argument('--radius', type=float, help='Determines spacing of particle picks')
parser.add_argument('--output_dir', type=str, help='Path of folder where picks are placed')

args = parser.parse_args()
DAE_DIR_PATH = args.dae_net_dir              #'./semSeg_50ktrain_catCrossEnt.h5'
SEMSEG_DIR_PATH = args.semseg_net_dir              #'./semSeg_50ktrain_catCrossEnt.h5'
BINNED_MICRO_PATH = args.binned_micro_dir          #'../Micrographs_bin4/*_bin4.mrc'
HIST_MATCH_PATH = args.hist_match_dir              #'./actin_rotated%05d.mrc'
INCREMENT = int(args.increment)                    #48
BOX_SIZE = int(args.box_size)                      #96
FUDGE_FAC = float(args.fudge_fac)                    #0.9
THRESHOLD = float(args.threshold)                    #192
GPU_IDX = int(args.gpu_idx)
TOT_GPUS = int(args.tot_gpus)                          #36
OUTPUT_DIR_PATH = args.output_dir                  #should be like 'PickParticles/job005'

if(OUTPUT_DIR_PATH[-1] != '/'): OUTPUT_DIR_PATH = OUTPUT_DIR_PATH + '/'

print('All inputs have been entered properly...')
print('Setting up output directories...')
# make directories
try:
	if(not os.path.isdir(OUTPUT_DIR_PATH+'denoised/')): os.mkdir(OUTPUT_DIR_PATH+'denoised/')
except OSError as err:
	print(err)
try:
	if(not os.path.isdir(OUTPUT_DIR_PATH+'semSeg/')): os.mkdir(OUTPUT_DIR_PATH+'semSeg/')
except OSError as err:
	print(err)
try:
	if(not os.path.isdir(OUTPUT_DIR_PATH+'starFiles/')): os.mkdir(OUTPUT_DIR_PATH+'starFiles/')
except OSError as err:
	print(err)

print('The program will now run.')
################################################################################
print('Python packages loaded. Setting CUDA environment...')
#GPU_IDX = 2
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_IDX%4)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

################################################################################
################################################################################
# load trained Fully Convolutional Network for semantic segmentation
################################################################################
print('Loading neural network models'); print('')
DAE = keras.models.load_model(DAE_DIR_PATH, custom_objects={'CCC':CCC})
FCN = keras.models.load_model(SEMSEG_DIR_PATH)

print('Network loaded')
# Load one test image for histogram matching
#hist_match_dir = './'
with mrcfile.open(HIST_MATCH_PATH) as mrc:
	hist_matcher = mrc.data

################################################################################
# Load real micrographs
#if('.txt' in BINNED_MICRO_PATH):
#	with open(BINNED_MICRO_PATH) as file:
#		file_names = file.readlines()
#		file_names = [line.rstrip() for line in file_names]

#else:
#	file_names = sorted(glob.glob(BINNED_MICRO_PATH+'*.mrc'))


# comment out this block to do full dataset
#import random 
#random.seed(4)
#file_names = random.sample(file_names, 40)
file_names = sorted(glob.glob(BINNED_MICRO_PATH+'*.mrc'))
#img_num = len(file_names)#len(sorted(glob.glob('../Micrographs_bin4/*_bin4.mrc')))
#img_start = int((img_num / 4)*GPU_IDX)
#img_end = int((img_num / 4)*(GPU_IDX+1))
#file_names = file_names[img_start:img_end]
file_names = np.array_split(np.array(file_names),TOT_GPUS)[GPU_IDX]
print(file_names)

# Do picks!
print('Predicting on files from ' + file_names[0] + ' to ' + file_names[-1])
for i in tqdm(range(0, len(file_names)), file=sys.stdout):
#for i in tqdm(range(0, 1), file=sys.stdout):
	try:
		temp = run_pick_on_micrograph(file_names[i],  INCREMENT, BOX_SIZE, hist_matcher, FUDGE_FAC, OUTPUT_DIR_PATH, DAE, FCN, THRESHOLD)
	except KeyboardInterrupt:
		print('Script hung on micrograph: ' + file_names[i] + ' ...')
		print('Proceeding to next micrograph.')



print('Exiting...')







