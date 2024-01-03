#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
################################################################################
# import of python packages
print('Beginning to import packages...')
import numpy as np
import matplotlib.pyplot as plt
import keras
import mrcfile
import random
from tqdm import tqdm
from keras import layers
from keras.models import Model
import tensorflow as tf; import keras.backend as K
import glob
import os
print('Packages finished importing.')
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

################################################################################
print('Checking for GPUs...')
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print('Loading neural networks...')
DAE_model_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/testing_backproject_corrWedge_adv/adversarial_trained_DAE_epoch_000.h5'
trained_DAE = keras.models.load_model(DAE_model_path, custom_objects={'CCC':CCC})

#FCN_model_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/FCN_training_epoch000_try2/fascin_CCE_0.0590_epoch_03.h5'
FCN_model_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/FCN_training_epoch000_try2/semSeg_50ktrain_catCrossEnt.h5'
trained_FCN = keras.models.load_model(FCN_model_path)
print('Loaded neural networks. Loading data...')

################################################################################
# Load data
real_data_path = '/rugpfs/fs0/cem/store/khamilton/tomos_for_matt/ts044.st_rec.mrc'

picks = [
	[88,1176,464],
	[88,762,497],
	[60,336,661],
	[60,500,451] # orig = [60,497,451]
]

data_holder = []
for i in tqdm(range(0, len(picks))):
	with mrcfile.open(real_data_path) as mrc:
		noise_data = mrc.data
		noise_data = -1.0*noise_data[picks[i][0]-32:picks[i][0]+32,picks[i][1]-32:picks[i][1]+32,picks[i][2]-32:picks[i][2]+32]
		#noise_data = hist_match(noise_data, hist_matcher)

	data_holder.append(noise_data)

data_holder = np.asarray(data_holder)
#data_holder = np.moveaxis(np.asarray(data_holder), 0,-1)
#data_holder = (data_holder - np.mean(data_holder, axis=(0,1,2),keepdims=True)) / np.std(data_holder, axis=(0,1,2),keepdims=True)
for i in range(0, len(data_holder)):
	data_holder[i] = (data_holder[i]-np.mean(data_holder[i]))/np.std(data_holder[i])

print('All data loaded. Performing inference...')

# Do predictions
print(np.expand_dims(data_holder[:,0], axis=-1).shape)
denoised_vols =  trained_DAE.predict(np.expand_dims(1.7*data_holder, axis=-1).astype('float16'))
segmented_vols = trained_FCN.predict(np.expand_dims(1.7*data_holder, axis=-1).astype('float16'))

output_dir_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/FCN_training_epoch000_try2/predictions_real/'
print('Saving data...')
for i in tqdm(range(0, len(denoised_vols))):
	with mrcfile.new(output_dir_path+real_data_path.split('/')[-1][:-4]+'_rawNoisy_' + str(i).zfill(2)+'.mrc', overwrite=True) as mrc:
		mrc.set_data(data_holder[i].astype('float32'))
	with mrcfile.new(output_dir_path+real_data_path.split('/')[-1][:-4]+'_denoised_' + str(i).zfill(2)+'.mrc', overwrite=True) as mrc:
		mrc.set_data(denoised_vols[i,:,:,:,0].astype('float32'))
	with mrcfile.new(output_dir_path+real_data_path.split('/')[-1][:-4]+'_semMap_bg_' + str(i).zfill(2)+'.mrc', overwrite=True) as mrc:
		mrc.set_data(segmented_vols[i,:,:,:,0].astype('float32'))
	with mrcfile.new(output_dir_path+real_data_path.split('/')[-1][:-4]+'_semMap_actin_' + str(i).zfill(2)+'.mrc', overwrite=True) as mrc:
		mrc.set_data(segmented_vols[i,:,:,:,1].astype('float32'))
	with mrcfile.new(output_dir_path+real_data_path.split('/')[-1][:-4]+'_semMap_fascin_' + str(i).zfill(2)+'.mrc', overwrite=True) as mrc:
		mrc.set_data(segmented_vols[i,:,:,:,2].astype('float32'))

print('Finished predicting on data. Exiting...')

