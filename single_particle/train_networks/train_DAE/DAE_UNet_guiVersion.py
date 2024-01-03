#!/home/greg/.conda/envs/matt_TF/bin/python
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
import os
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Generate noisy and noiseless 2D projections of randomly oriented and translated MRC files.')
parser.add_argument('--input_noise_dir', type=str, help='output directory to store noisy 2D projections')
parser.add_argument('--input_noiseless_dir', type=str, help='output directory to store noiseless 2D projections')
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
if(args.input_noise_dir == None or args.input_noiseless_dir == None or args.proj_dim == None or 
	args.numProjs == None or args.tv_split == None or args.lr == None or args.patience == None or 
	args.epochs == None or args.batch_size == None):
	sys.exit('Please enter inputs correctly.')

if(args.gpu_idx == None):
	print('No GPU index specified, using first GPU.')
	GPU_IDX = str('1')
else:
	GPU_IDX = str(args.gpu_idx)

NOISY_DIR_PATH = args.input_noise_dir              #'/scratch/neural_network_training_sets/tplastin_noise/'
NOISELESS_DIR_PATH = args.input_noiseless_dir      #'/scratch/neural_network_training_sets/tplastin_noNoise/'
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
if(NOISELESS_DIR_PATH[-1] != '/'): NOISELESS_DIR_PATH = NOISELESS_DIR_PATH + '/'
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
		with mrcfile.open(noNoise_folder + file_name) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noNoise_data = mrc.data
				
		if(not np.isnan(noise_data).any() and not np.isnan(noNoise_data).any()): #doesn't have a nan
			noise_holder.append(noise_data.astype('float16'))
			noNoise_holder.append(noNoise_data.astype('float16'))
		
		else: # i.e. if mrc.data does have an nan, skip it and print a statement
			print('Training image number %d has at least one nan value. Skipping this image.'%i)
	
	return noise_holder, noNoise_holder

################################################################################
#https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
import keras.backend as K
def custom_loss(weights, outputs):
	def contractive_loss(y_pred, y_true):
		lam = 1e-2
		#print(len(autoencoder.layers))
		mse = K.mean(K.square(y_true - y_pred), axis=1)
		W = K.variable(value=weights)  # N x N_hidden
		W = K.transpose(W)  # N_hidden x N
		h = outputs
		dh = h * (1 - h)  # N_batch x N_hidden
		# N_batch x N_hidden * N_hidden x 1 = N_batch x 1
		contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
		return mse + contractive
	
	return contractive_loss

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
noise_folder = NOISY_DIR_PATH#folder + 'tplastin_noise/'#'noise_proj4/'
noNoise_folder = NOISELESS_DIR_PATH#folder + 'tplastin_noNoise/'#'noNoise_proj4/'

train, target = import_synth_data(noise_folder, noNoise_folder, BOX_DIM, 0, NUM_NOISE_PAIRS)
train = np.asarray(train, dtype='float16'); target = np.asarray(target,dtype='float16')

#add extra dimension at end because only one color channel
train = np.expand_dims(train, axis=-1)
target = np.expand_dims(target, axis=-1)

FRAC_VAL = int(train.shape[0] * TV_SPLIT)
val_train = train[:FRAC_VAL]
val_target = target[:FRAC_VAL]
train = train[FRAC_VAL:]
target = target[FRAC_VAL:]
print('All files loaded and parsed into training and validation sets.')
print('Beginning training')

################################################################################
######### The data should be imported; now build the model #####################
################################################################################
# Define the model
def create_model_dense(training_data, full_training, lr):
	# Instantiate the model
	input_img = layers.Input(shape=(training_data.shape[1:]))
	# Make layers
	x = layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu',trainable=full_training)(input_img) #[192x192x10]
	x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x192 = layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x192)#[96x96x16]
	x = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv2D(168, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x96 = layers.Conv2D(168, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x96)#[48x48x64]
	x = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv2D(192, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x48 = layers.Conv2D(192, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	
	x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x48)#[24x24x128]
	x = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[24x48x128]
	x = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[24x48x128]
	x = layers.Conv2D(256, kernel_size=(1,1), activation='relu', padding='same',trainable=full_training)(x)#[24x48x128]
	x = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same',trainable=full_training)(x)#[24x48x128]
	
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
	decoded = layers.Conv2D(1, (1,1), activation='linear', padding='same',trainable=full_training)(x)#40
	
	# optimizer
	adam = keras.optimizers.Adam(lr=lr)
	# Compile model
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer=adam, loss=CCC, metrics=['mse'])
	#autoencoder.compile(optimizer=adam, loss='mse', metrics=[CCC])
	autoencoder.summary()
	return autoencoder, x

################################################################################
# Handle model
def train_model(train_data, train_target):
	autoencoder, encoder = create_model_dense(train_data,True, LEARNING_RATE)
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=PATIENCE, restore_best_weights=True)
	checkpoint = keras.callbacks.ModelCheckpoint(OUTPUT_DIR_PATH + '300ktraining_tplastin_CCC00000_best_model_so_far_fig.h5', monitor='loss', verbose=0, save_best_only=True, mode='auto',save_weights_only=False, period=True)
	history = autoencoder.fit(x=train_data, y=train_target, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es,checkpoint])
	return [autoencoder, history, encoder]

def continue_training(train_data, train_target, epochs, autoencoder_model):
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2, restore_best_weights=True)
	history = autoencoder_model.fit(x=train_data, y=train_target, epochs=epochs, batch_size=BATCH_SIZE, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es])
	return [autoencoder_model, history]

################################################################################
# train the three models. First do greedy training of outer layers then inner layers
# then train full model
autoencoder_three, history_three, encoder_three = train_model(train,target)

# continue training, if needed
#autoencoder_longer_train, history_longer_train = continue_training(train[:100000], target[:100000], 2, autoencoder_three)

# save the final model
model_save_name = OUTPUT_DIR_PATH + 'fig_300000training_tplastin_CCC00000.h5'
print('Model finished training.\nSaving model as ' + model_save_name)
autoencoder_three.save(model_save_name)

import pickle
with open(OUTPUT_DIR_PATH + 'DAE_trainHistoryDict_fig', 'wb') as file_pi:
	pickle.dump(history_three.history, file_pi)

print('Finished everything.')
print('Exiting...')


'''plot_history(history_three)

################################################################################
################################################################################
# check conv-dense autoencoder
check_num = 25
cm = plt.get_cmap('gray')#plt.cm.greens
predict_conv = autoencoder_three.predict(np.expand_dims(train[check_num].astype('float16'), axis=0))[0,:,:,0]
predict_dense = -1.0*autoencoder_three.predict(np.expand_dims(train[check_num].astype('float16'), axis=0))[0,:,:,0]
fig,ax = plt.subplots(2,2); _=ax[0,0].imshow(train[check_num,:,:,0].astype('float32'), cmap=cm); _=ax[0,1].imshow(target[check_num,:,:,0].astype('float32'), cmap=cm); _=ax[1,0].imshow(predict_conv.astype('float32'), cmap=cm);_=ax[1,1].imshow(predict_dense,cmap=cm);  #plt.show(block=False)

#encoder_model = Model(autoencoder_three.input, autoencoder_three.layers[21].output)
#encoded_pred = encoder_model.predict(np.expand_dims(train[check_num].astype('float16'), axis=0))[0]
#ax[0,2].plot(encoded_pred);
plt.show()



with mrcfile.new('noisy_training%02d.mrc'%i, overwrite=True) as mrc:
	mrc.set_data(train[check_num][:,:,0].astype('float32'))

with mrcfile.new('ground_truth%02d.mrc'%i, overwrite=True) as mrc:
	mrc.set_data(target[check_num][:,:,0].astype('float32'))

with mrcfile.new('predicted_synth%02d.mrc'%i, overwrite=True) as mrc:
	mrc.set_data(predict_dense.astype('float32'))


################################################################################
################################################################################
# if you want to see learning curves, plot this
def plot_history(history):
	p1, = plt.plot(history.history['loss']); p2, = plt.plot(history.history['val_loss']); 
	plt.title('Loss'); plt.ylim(ymin=-1); 
	plt.legend((p1,p2), ('Training Loss', 'Validation Loss'), loc='upper right', shadow=True)
	plt.show()

################################################################################
'''
