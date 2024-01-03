#!/home/greg/.conda/envs/matt_TF/bin/python
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
from scipy import interpolate; from scipy.ndimage import filters
import os
print('Packages finished importing. Data will now be loaded')
################################################################################
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Generate noisy and noiseless 2D projections of randomly oriented and translated MRC files.')
parser.add_argument('--input_noise_dir', type=str, help='output directory to store noisy 2D projections')
parser.add_argument('--input_semseg_dir', type=str, help='output directory to store noiseless 2D projections')
parser.add_argument('--trained_DAE_path', type=str, help='output directory to store noisy 2D projections')
parser.add_argument('--proj_dim', type=str, help='output directory to store semantic segmentation targets')
parser.add_argument('--numProjs', type=int, help='total number of projections to make')
parser.add_argument('--tv_split', type=int, help='total number of parallel threads to launch')
parser.add_argument('--lr', type=float, help='Box size of projected image')
parser.add_argument('--patience', type=int, help='Box size of projected image')
parser.add_argument('--epochs', type=int, help='Box size of projected image')
parser.add_argument('--batch_size', type=int, help='Box size of projected image')
parser.add_argument('--gpu_idx', type=int, help='Box size of projected image')
parser.add_argument('--output_dir', type=str, help='Box size of projected image')

args = parser.parse_args()
print('')
if(args.input_noise_dir == None or args.input_semseg_dir == None or args.proj_dim == None or 
	args.numProjs == None or args.tv_split == None or args.lr == None or args.patience == None or 
	args.epochs == None or args.batch_size == None):
	sys.exit('Please enter inputs correctly.')

if(args.gpu_idx == None):
	print('No GPU index specified, using first GPU.')
	GPU_IDX = str('1')
else:
	GPU_IDX = str(args.gpu_idx)

NOISY_DIR_PATH = args.input_noise_dir              #'/scratch/neural_network_training_sets/tplastin_noise/'
SEMSEG_DIR_PATH = args.input_semseg_dir      #'/scratch/neural_network_training_sets/tplastin_noNoise/'
TRAINED_DAE_PATH = args.trained_DAE_path
BOX_DIM = 192#int(args.proj_dim)                       #192
NUM_NOISE_PAIRS = int(args.numProjs)               #1000
LEARNING_RATE = float(args.lr)                     #0.00005
PATIENCE = int(args.patience)                      #3
EPOCHS = int(args.epochs)                          #10
BATCH_SIZE = int(args.batch_size)                  #16
TV_SPLIT = (100.0-float(args.tv_split))/100.0      #90
OUTPUT_DIR_PATH = args.output_dir                  #should be like 'TrainNetworks/job005'

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_IDX

if(NOISY_DIR_PATH[-1] != '/'): NOISY_DIR_PATH = NOISY_DIR_PATH + '/'
if(SEMSEG_DIR_PATH[-1] != '/'): SEMSEG_DIR_PATH = SEMSEG_DIR_PATH + '/'
if(OUTPUT_DIR_PATH[-1] != '/'): OUTPUT_DIR_PATH = OUTPUT_DIR_PATH + '/'


print('All inputs have been entered properly. The program will now run.')

################################################################################
# method to import synthetic data from files
def import_synth_data(noise_folder, noNoise_folder, box_length, NUM_IMGS_MIN, NUM_IMGS_MAX):
	noise_holder = []; noNoise_holder = []
	print('Loading files from ' + noise_folder)
	for i in tqdm(range(NUM_IMGS_MIN, NUM_IMGS_MAX), file=sys.stdout):
		file_name = 'actin_rotated%05d.mrc'%i
		noise_data = None; noNoise_data = None
		with mrcfile.open(noise_folder + file_name) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noise_data = mrc.data
		with mrcfile.open(noNoise_folder + file_name + 's') as mrc:
			#if(mrc.data.shape == (box_length,box_length)):
			noNoise_data = mrc.data
				
		if(not np.isnan(noise_data).any() and not np.isnan(noNoise_data).any()): #doesn't have a nan
			noise_holder.append(noise_data.astype('float16'))
			noNoise_holder.append(noNoise_data.astype('float16'))
		
		else: # i.e. if mrc.data does have an nan, skip it and print a statement
			print('Training image number %d has at least one nan value. Skipping this image.'%i)
	
	return noise_holder, noNoise_holder

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
	return -1*r

################################################################################
#folder = '/scratch/neural_network_training_sets/'
noise_folder = NOISY_DIR_PATH#folder + 'tplastin_noise/'#
noNoise_folder = SEMSEG_DIR_PATH#folder + 'tplastin_semMaps/'#

train, target = import_synth_data(noise_folder, noNoise_folder, BOX_DIM, 0, NUM_NOISE_PAIRS)
train = np.asarray(train, dtype='float16'); target = np.asarray(target,dtype='float16')

#add extra dimension at end because only one color channel
train = np.expand_dims(train, axis=-1)
target = np.moveaxis(target, 1, -1)

FRAC_VAL = int(train.shape[0] * TV_SPLIT)
val_train = train[:FRAC_VAL]
val_target = target[:FRAC_VAL]
train = train[FRAC_VAL:]
target = target[FRAC_VAL:]
print('All files loaded and parsed into training and validation sets.')
print('Beginning training')

################################################################################
######### The data should be imported; now create the model ####################
################################################################################
# Import the encoding layers of the DAE model
model_path = TRAINED_DAE_PATH#'../300000training_tplastin_CCC09856.h5'
trained_DAE = keras.models.load_model(model_path, custom_objects={'CCC':CCC})

################################################################################
# Define the model
def create_model_dense(training_data, full_training, lr):
	input_img = layers.Input(shape=(training_data.shape[1:]))
	x = layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu',trainable=False)(input_img) #[192x192x10]
	x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[192x192x16]
	x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[192x192x16]
	x = layers.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same',trainable=False)(x)#[192x192x16]
	x192 = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[192x192x16]
	
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x192)#[96x96x16]
	x = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[96x96x64]
	x = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[96x96x64]
	x = layers.Conv2D(168, kernel_size=(1,1), activation='relu', padding='same',trainable=False)(x)#[96x96x64]
	x96 = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[96x96x64]
	
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x96)#[48x48x64]
	x = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[48x48x128]
	x = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[48x48x128]
	x = layers.Conv2D(192, kernel_size=(1,1), activation='relu', padding='same',trainable=False)(x)#[48x48x128]
	x48 = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[48x48x128]
	
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x48)#[24x24x128]
	x = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[24x48x128]
	x = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[24x48x128]
	x = layers.Conv2D(256, kernel_size=(1,1), activation='relu', padding='same',trainable=False)(x)#[24x48x128]
	x = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same',trainable=False)(x)#[24x48x128]
	
	x = layers.UpSampling2D((2,2))(x)#[48x48x128]
	x = layers.Concatenate(axis=-1)([x, x48])#[48x48x256]
	x = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv2D(192, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	
	x = layers.UpSampling2D((2,2))(x)#[96x96x128]
	x = layers.Concatenate(axis=-1)([x, x96])#[192x192x192]
	x = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv2D(168, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	
	x = layers.UpSampling2D((2,2))(x)#[192x192x64]
	x = layers.Concatenate(axis=-1)([x, x192])#[192x192x80]
	x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)
	decoded = layers.Conv2D(3, (1,1), activation='softmax', padding='same',trainable=full_training)(x)#40
	
	# optimizer
	adam = keras.optimizers.Adam(lr=lr)
	# Compile model
	semSeg = Model(input_img, decoded)
	semSeg.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
	semSeg.summary()
	return semSeg


semSeg = create_model_dense(train,True, 0.00001) #0.00005
for i in range(0, len(trained_DAE.layers)-1):
	semSeg.layers[i].set_weights(trained_DAE.layers[i].get_weights())

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=PATIENCE, restore_best_weights=True)
history = semSeg.fit(x=train, y=target, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es])


model_save_name = OUTPUT_DIR_PATH + 'semSeg_50ktrain_catCrossEnt_fig.h5'
print('Model finished training.\nSaving model as ' + model_save_name)
semSeg.save(model_save_name)
print('Model saved.')



import pickle
with open(OUTPUT_DIR_PATH + 'semSeg_trainHistoryDict_fig', 'wb') as file_pi:
	pickle.dump(history.history, file_pi)

print('Training history saved.')
print('Exiting...')

################################################################################
################################################################################
"""
check_num = 9
cm = plt.get_cmap('gray')#plt.cm.greens
predict_conv = semSeg.predict(np.expand_dims(train[check_num].astype('float16'), axis=0))[0]
fig,ax = plt.subplots(3,3); 
_=ax[0,0].imshow(train[check_num,:,:,0].astype('float32'), cmap=cm); 

_=ax[1,0].imshow(target[check_num,:,:,0].astype('float32'), cmap=cm); 
_=ax[1,1].imshow(target[check_num,:,:,1].astype('float32'), cmap=cm); 
_=ax[1,2].imshow(target[check_num,:,:,2].astype('float32'), cmap=cm);
 
_=ax[2,0].imshow(predict_conv.astype('float32')[:,:,0], cmap=cm);
_=ax[2,1].imshow(predict_conv.astype('float32')[:,:,1], cmap=cm);
_=ax[2,2].imshow(predict_conv.astype('float32')[:,:,2], cmap=cm);
plt.show()

"""




