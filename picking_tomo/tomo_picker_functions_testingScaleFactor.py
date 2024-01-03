#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker/bin/python
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
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter
import threading
try:
	import thread
except ImportError:
	import _thread as thread

import argparse; import sys
################################################################################
def quit_function(fn_name):
	# print to stderr, unbuffered in Python 2.
	#print('{0} took too long'.format(fn_name), file=sys.stderr)
	#sys.stderr.flush() # Python 3 stderr is likely buffered.
	thread.interrupt_main() # raises KeyboardInterrupt

def exit_after(s):
	'''
	use as decorator to exit process if 
	function takes longer than s seconds
	'''
	def outer(fn):
		def inner(*args, **kwargs):
			timer = threading.Timer(s, quit_function, args=[fn.__name__])
			timer.start()
			try:
				result = fn(*args, **kwargs)
			finally:
				timer.cancel()
			return result
		return inner
	return outer

################################################################################
################################################################################
################# Functions for operating on whole micrographs ################
################################################################################
################################################################################
def CCC(y_pred, y_true):
	x = y_true
	y = y_pred
	mx=K.mean(x)
	my=K.mean(y)
	xm, ym = x-mx, y-my
	r_num = K.sum(tf.multiply(xm,ym))
	r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
	r = r_num / r_den
	#return -1*r
	if tf.math.is_nan(r):
		return tf.cast(1, tf.float16)
	else:
		return tf.cast(-1*r, tf.float16)




def slice_up_micrograph(real_data, increment_x, increment_y, increment_z, box_size,hist_matcher):#, box_size):
	extractions = []
	for i in tqdm(range(0, real_data.shape[0]-box_size, increment_z)):
		for j in range(0, real_data.shape[1]-box_size, increment_y):
			for k in range(0, real_data.shape[2]-box_size, increment_x):
				extraction = -1.0*real_data[i:i+box_size,j:j+box_size,k:k+box_size]
				#extraction = hist_match(extraction, hist_matcher)
				# rotate extraction by 90 degrees because training data was rotated wrong
				#extraction = np.rot90(extraction, k=1, axes=(1,2))
				extractions.append(extraction)
	extractions = np.moveaxis(np.asarray(extractions), 0,-1)
	print(extractions.shape)
	print('Normalizing extractions...')
	extractions = (extractions - np.mean(extractions, axis=(0,1,2),keepdims=True)) / np.std(extractions, axis=(0,1,2),keepdims=True)
	return np.expand_dims(np.moveaxis(extractions, -1, 0), axis=-1).astype('float16')

def stitch_back_seg(shape, preds, increment_x, increment_y, increment_z, box_size):
	stitch_back = np.zeros((shape))
	cntr=0
	for i in range(0, stitch_back.shape[0]-box_size, increment_z):
		for j in range(0, stitch_back.shape[1]-box_size, increment_y):
			for k in range(0, stitch_back.shape[2]-box_size, increment_x):
				#rotated_back = np.rot90(preds[cntr], k=3, axes=(1,2))
				stitch_back[i:i+box_size, j:j+box_size, k:k+box_size] = np.max(np.stack((preds[cntr], stitch_back[i:i+box_size, j:j+box_size, k:k+box_size])),axis=0)
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


@exit_after(200000)
def run_pick_on_micrograph(file_name, INCREMENT, BOX_SIZE, hist_matcher, FUDGE_FAC, OUTPUT_DIR_PATH, DAE, FCN, THRESHOLD):
	print('Predicting on Tomogram...: '+file_name)
	big_micrograph_name = file_name
	with mrcfile.open(big_micrograph_name) as mrc:
		real_data = mrc.data
	
	print(real_data.shape)
	
	real_data = real_data[50:140,200:1300,300:750]
	################################################################################
	# Divide up the whole micrograph to feed to the FCN for semantic segmentation
	#increment = 48
	print('Slicing up tomogram now...')
	extractions = slice_up_micrograph(real_data, INCREMENT, INCREMENT, 4, BOX_SIZE, hist_matcher)
	possible_scales = [0.8,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
	for j in range(0, len(possible_scales)):
		extractions_copy = possible_scales[j]*extractions
		print('Extraction shape: '+ str(extractions.shape))
		print('Running DAE prediction now...')
		zero_box = np.zeros((8,64,64,64,1))
		pad = 8
		zero_box[:,pad:-pad,pad:-pad,pad:-pad] = 1
		dae_preds = []
		for i in tqdm(range(0, extractions.shape[0], 8)):
			batch_pred = DAE.predict(extractions_copy[i:i+8])
			batch_pred = np.multiply(zero_box[:len(batch_pred)], batch_pred)
			dae_preds.append(batch_pred)
		
		dae_preds = np.concatenate(dae_preds, axis=0)[:,:,:,:,0]
		print(dae_preds.shape)
		print('Stitching tomogram back together...')
		dae_stitch_back = stitch_back_seg(real_data.shape, dae_preds, INCREMENT, INCREMENT, 4, BOX_SIZE)
		
		with mrcfile.new(OUTPUT_DIR_PATH+'denoised/scale_%s'%str(possible_scales[j])+big_micrograph_name[:-4].split('/')[-1]+'.mrc', overwrite=True) as mrc:
			#mrc.set_data((stitch_back>THRESHOLD).astype('float32')) # binarized
			mrc.set_data((dae_stitch_back).astype('float32')) #Non-binarized
		
	'''
	print('Running FCN prediction now...')
	fcn_preds = []
	for i in tqdm(range(0, extractions.shape[0], 8)):
		batch_pred = FCN.predict(extractions[i:i+8])
		fcn_preds.append(batch_pred)
	
	fcn_preds = np.concatenate(fcn_preds, axis=0)
	fcn_preds_channel_0 = fcn_preds[:,:,:,:,0]
	fcn_preds_channel_1 = fcn_preds[:,:,:,:,1]
	fcn_preds_channel_2 = fcn_preds[:,:,:,:,2]
	print('Finished making predictions. Stitching back segmented tomograms...')
	stitch_back_channel_0 = stitch_back_seg(real_data.shape, fcn_preds_channel_0, INCREMENT, INCREMENT, 4, BOX_SIZE)
	stitch_back_channel_1 = stitch_back_seg(real_data.shape, fcn_preds_channel_1, INCREMENT, INCREMENT, 4, BOX_SIZE)
	stitch_back_channel_2 = stitch_back_seg(real_data.shape, fcn_preds_channel_2, INCREMENT, INCREMENT, 4, BOX_SIZE)
	print('Saving segmented tomograms...')
	with mrcfile.new(OUTPUT_DIR_PATH+'semSeg/'+big_micrograph_name[:-4].split('/')[-1]+'_channel0.mrc', overwrite=True) as mrc:
		#mrc.set_data((stitch_back>THRESHOLD).astype('float32')) # binarized
		mrc.set_data((stitch_back_channel_0).astype('float32')) #Non-binarized
	
	with mrcfile.new(OUTPUT_DIR_PATH+'semSeg/'+big_micrograph_name[:-4].split('/')[-1]+'_channel1.mrc', overwrite=True) as mrc:
		#mrc.set_data((stitch_back>THRESHOLD).astype('float32')) # binarized
		mrc.set_data((stitch_back_channel_1).astype('float32')) #Non-binarized
	
	with mrcfile.new(OUTPUT_DIR_PATH+'semSeg/'+big_micrograph_name[:-4].split('/')[-1]+'_channel2.mrc', overwrite=True) as mrc:
		#mrc.set_data((stitch_back>THRESHOLD).astype('float32')) # binarized
		mrc.set_data((stitch_back_channel_2).astype('float32')) #Non-binarized
	
	'''
	'''
	triple_pt_box_size = 16
	num_to_prune = 8
	skel_pruned = skeletonize_and_prune_nubs(stitch_back, triple_pt_box_size, num_to_prune, THRESHOLD)
	
	# Now plot end points to see results
	E_img = filters.convolve(skel_pruned,[[1,1,1],[1,10,1],[1,1,1]])
	end_pts = np.asarray(np.argwhere(E_img==11))
	################################################################################
	# Now that we have a clean, image-wide skeleton; do instance segmentation
	filaments = define_filaments_from_skelPruned(skel_pruned, end_pts, 10)
	# Plot whole micrograph with segmented pixels
	real_data = -1.0*real_data
	real_data = (real_data - np.mean(real_data)) / np.std(real_data)
	_=plt.imshow(-1.0*real_data, cmap=plt.cm.gray)
	colors = iter(plt.cm.rainbow(np.linspace(0,1,len(filaments))))
	rs = []; curvatures = []
	for j in range(0, len(filaments)):
		_=plt.scatter(filaments[j][:,1], filaments[j][:,0], s=0.1, alpha=0.2, color=next(colors))
		curvature, r = spline_curv_estimate(filaments[j], len(filaments[j]), 10, 2, 3) # sampling(usually10), buff, min_threshold
		if(curvature != [] and len(curvature) > 3):
			r_curv_range = r[np.argwhere(np.logical_and(-10e20<curvature, curvature<10e20))][:,0,:]
			kept_curvs = curvature[np.argwhere(np.logical_and(-10e20<curvature, curvature<10e20))]
			plt.scatter(r_curv_range[:,1], r_curv_range[:,0], c='red',s=5, alpha=0.5)
			rs.append(np.concatenate((r_curv_range,kept_curvs),axis=-1)); curvatures.append(curvature)
	
	plt.tight_layout()
	plt.savefig(OUTPUT_DIR_PATH+'pngs/'+big_micrograph_name[:-4].split('/')[-1]+'.png', dpi=600)
	plt.clf()
	
	################################################################################
	# Extract and predict rotation
	step_size_px = 1
	rs_full = []; extractions_full = []
	for j in range(0, len(rs)):
		if(rs[j].shape[0] != 0 and rs[j].shape[0]>1):
			rot_angles = np.zeros((rs[j].shape[0], 2))
			extraction = extract_filaments(real_data, hist_matcher, rs[j], step_size_px)
			xyrot = np.concatenate((rs[j], rot_angles), axis=1)
			rs_full.append(xyrot)
			extractions_full.append(extraction)
	
	
	curv_preds = []
	for j in range(0, len(rs_full)):
		measured_curv = 0#curv_measurement.predict(np.expand_dims(extractions_full[j], axis=-1))
		rs_full[j][:,3] = 0#measured_curv[:,0] #curvature
		curv_preds.append(measured_curv)
	
	
	################################################################################
	# Prepare star file
	header = '# RELION; version 3.0\n\ndata_\n\nloop_ \n_rlnCoordinateX #1 \n_rlnCoordinateY #2 \n_rlnAngleTiltPrior #3 \n'
	star_file = header
	for j in range(0, len(rs_full)):
		for k in range(0, len(rs_full[j])):
			star_file = star_file + starify(rs_full[j][k][1]*4.0,rs_full[j][k][0]*4.0,90.0)
	
	star_path = OUTPUT_DIR_PATH+'starFiles/'+(big_micrograph_name[:-4].split('/')[-1]).split('_bin4')[0]
	with open(star_path+'.star', "w") as text_file:
		text_file.write(star_file)
	'''






