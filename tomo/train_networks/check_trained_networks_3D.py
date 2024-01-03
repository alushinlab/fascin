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

################################################################################
print('Checking for GPUs...')
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print('Loading neural networks...')
#DAE_model_path = './result_trained_networks/fascinTomo_tester_CCC_-0.8281_epoch_03.h5'
#DAE_model_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/result_trained_networks/fascin_tomo_tester_box64_CCC0p9846.h5'
#DAE_model_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/testing_backproject_adv/adversarial_trained_DAE_epoch_002.h5'
DAE_model_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/testing_backproject_corrWedge_adv/adversarial_trained_DAE_epoch_000.h5'
trained_DAE = keras.models.load_model(DAE_model_path, custom_objects={'CCC':CCC})

#FCN_model_path = './result_trained_networks/fascinTomo_BCE_0.0930_epoch_05.h5'
FCN_model_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/FCN_training_epoch000_try2/fascin_CCE_0.0590_epoch_03.h5'
trained_FCN = keras.models.load_model(FCN_model_path)
print('Loaded neural networks. Loading data...')

################################################################################
# Load data
synth_data_path = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject_rot90/noise_dir/*.mrc'))[-20:-10]
gt_noiseless_data_path = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject_rot90/noiseless_dir/*.mrc'))[-20:-10]
gt_semMap_data_path_actin = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject_rot90/semMap/actin*.mrc'))[-20:-10]
gt_semMap_data_path_fascin = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject_rot90/semMap/fascin*.mrc'))[-20:-10]
gt_semMap_data_path_background = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/generate_synth_data_3d/test_backproject_rot90/semMap/background*.mrc'))[-20:-10]

data_holder = []
for i in tqdm(range(0, len(synth_data_path))):
	with mrcfile.open(synth_data_path[i]) as mrc:
		noise_data = mrc.data
	with mrcfile.open(gt_noiseless_data_path[i]) as mrc:
		noiseless_data = mrc.data
	with mrcfile.open(gt_semMap_data_path_actin[i]) as mrc:
		actinSemMap_data = mrc.data
	with mrcfile.open(gt_semMap_data_path_fascin[i]) as mrc:
		fascinSemMap_data = mrc.data
	with mrcfile.open(gt_semMap_data_path_background[i]) as mrc:
		backgroundSemMap_data = mrc.data
	
	data_holder.append([noise_data, noiseless_data, actinSemMap_data, fascinSemMap_data, backgroundSemMap_data])

data_holder = np.asarray(data_holder)

print('All data loaded. Performing inference...')

# Do predictions
print(np.expand_dims(data_holder[:,0], axis=-1).shape)
denoised_vols =  trained_DAE.predict(np.expand_dims(data_holder[:,0], axis=-1).astype('float16'))
segmented_vols = trained_FCN.predict(np.expand_dims(data_holder[:,0], axis=-1).astype('float16'))

output_dir_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/particle_picking/FCN_training_epoch000_try2/predictions_synth/'
print('Saving data...')
for i in tqdm(range(0, len(denoised_vols))):
	with mrcfile.new(output_dir_path+'noisy_' + synth_data_path[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(data_holder[i][0].astype('float32'))
	with mrcfile.new(output_dir_path+'noiseless_gt_' + gt_noiseless_data_path[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(data_holder[i][1].astype('float32'))
	with mrcfile.new(output_dir_path+'semMap_gt_' + gt_semMap_data_path_actin[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(data_holder[i][2].astype('float32'))
	with mrcfile.new(output_dir_path+'semMap_gt_' + gt_semMap_data_path_fascin[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(data_holder[i][3].astype('float32'))
	with mrcfile.new(output_dir_path+'semMap_gt_' + gt_semMap_data_path_background[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(data_holder[i][4].astype('float32'))
	with mrcfile.new(output_dir_path+'denoised_' + synth_data_path[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(denoised_vols[i,:,:,:,0].astype('float32'))
	with mrcfile.new(output_dir_path+'semMap_' + gt_semMap_data_path_actin[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(segmented_vols[i,:,:,:,0].astype('float32'))
	with mrcfile.new(output_dir_path+'semMap_' + gt_semMap_data_path_fascin[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(segmented_vols[i,:,:,:,1].astype('float32'))
	with mrcfile.new(output_dir_path+'semMap_' + gt_semMap_data_path_background[i].split('/')[-1], overwrite=True) as mrc:
		mrc.set_data(segmented_vols[i,:,:,:,2].astype('float32'))

print('Finished predicting on data. Exiting...')

