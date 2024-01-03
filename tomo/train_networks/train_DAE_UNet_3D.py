#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
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
import glob
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Generate noisy and noiseless 2D projections of randomly oriented and translated MRC files.')
parser.add_argument('--input_noise_dir', type=str, help='output directory to store noisy 2D projections')
parser.add_argument('--input_noiseless_dir', type=str, help='output directory to store noiseless 2D projections')
parser.add_argument('--proj_dim', type=str, help='output directory to store semantic segmentation targets')
parser.add_argument('--numProjs', type=int, help='total number of projections to load')
parser.add_argument('--tv_split', type=int, help='Training/validation split 90 means train on 90% validate on 10%')
parser.add_argument('--lr', type=float, help='Learning rate; 0.00005 usually works well')
parser.add_argument('--patience', type=int, help='Wait this many generations with no improvement before converging')
parser.add_argument('--epochs', type=int, help='Epochs to train')
parser.add_argument('--batch_size', type=int, help='Batch size; 16 is a good number to start with')
parser.add_argument('--gpu_idx', type=int, help='Which GPU index to train on')
parser.add_argument('--preload_ram', type=str, help='Pre-load all training data to RAM? If yes, type True')
parser.add_argument('--output_dir', type=str, help='Directory to save neural network')

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
BOX_DIM = int(args.proj_dim)                       #192
NUM_NOISE_PAIRS = int(args.numProjs)               #1000
LEARNING_RATE = float(args.lr)                     #0.00005
PATIENCE = int(args.patience)                      #3
EPOCHS = int(args.epochs)                          #10
BATCH_SIZE = int(args.batch_size)                  #16
TV_SPLIT = (100.0-float(args.tv_split))/100.0      #90
OUTPUT_DIR_PATH = args.output_dir                  #should be like 'TrainNetworks/job005'
PRELOAD_RAM = (args.preload_ram == 'True') or (args.preload_ram == 'true')        #


os.environ["CUDA_VISIBLE_DEVICES"]=GPU_IDX
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


if(NOISY_DIR_PATH[-1] != '/'): NOISY_DIR_PATH = NOISY_DIR_PATH + '/'
if(NOISELESS_DIR_PATH[-1] != '/'): NOISELESS_DIR_PATH = NOISELESS_DIR_PATH + '/'
if(OUTPUT_DIR_PATH[-1] != '/'): OUTPUT_DIR_PATH = OUTPUT_DIR_PATH + '/'


print('All inputs have been entered properly. The program will now run.')
################################################################################
# method to import synthetic data from files
def import_synth_data(noise_folder, noNoise_folder, box_length, NUM_IMGS_MAX):
	noise_holder = []; noNoise_holder = []
	print('Loading files from ' + noise_folder)
	noise_file_names = sorted(glob.glob(noise_folder+'*.mrc'))
	noNoise_file_names = sorted(glob.glob(noNoise_folder+'*.mrc'))
	if(len(noise_file_names) != len(noNoise_file_names)):
		print('Noise and noNoise file name lists not the same size. Quitting...')
		sys.exit()
	for i in tqdm(range(0, NUM_IMGS_MAX), file=sys.stdout):
		noise_data = None; noNoise_data = None
		with mrcfile.open(noise_file_names[i]) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noise_data = mrc.data
		with mrcfile.open(noNoise_file_names[i]) as mrc:
			if(mrc.data.shape == (box_length,box_length)):
				noNoise_data = mrc.data
				
		if(not np.isnan(noise_data).any() and not np.isnan(noNoise_data).any()): #doesn't have a nan
			noise_holder.append(noise_data.astype('float16'))
			noNoise_holder.append(noNoise_data.astype('float16'))
		
		else: # i.e. if mrc.data does have an nan, skip it and print a statement
			print('Training image number %d has at least one nan value. Skipping this image.'%i)
	
	return noise_holder, noNoise_holder


def mrc_generator(noisy_file_list, noiseless_file_list, box_length, start_idx, end_idx, batch_size=32):
	while True:
		# Loop over batches of files
		for i in range(start_idx, end_idx, batch_size):
			# Load the batch of files
			batch_files_noisy = noisy_file_list[i:i+batch_size]
			batch_files_noiseless = noiseless_file_list[i:i+batch_size]

			# Initialize the input and output arrays
			batch_x = np.zeros((len(batch_files_noisy), box_length, box_length, box_length, 1), dtype=np.float16)
			batch_y = np.zeros((len(batch_files_noiseless), box_length, box_length, box_length, 1), dtype=np.float16)
			
			# Loop over the files in the batch
			for j in range(0, len(batch_files_noisy)):
				# Read the MRC volume file
				with mrcfile.open(batch_files_noisy[j], mode='r', permissive=True) as mrc:
					noisy_data = mrc.data.astype(np.float16)
				with mrcfile.open(batch_files_noiseless[j], mode='r', permissive=True) as mrc:
					noiseless_data = mrc.data.astype(np.float16)
				
				if(np.isnan(noisy_data).any() or np.isnan(noiseless_data).any()):
					continue
				
				if(noisy_data.shape == (box_length,box_length,box_length) and noiseless_data.shape == (box_length,box_length,box_length)):
					# Add the volume to the input array
					batch_x[j, :, :, :, 0] = noisy_data
					batch_y[j, :, :, :, 0] = noiseless_data
			
			yield (batch_x, batch_y)



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
	#return -1*r
	if tf.math.is_nan(r):
		return tf.cast(1, tf.float16)
	else:
		return tf.cast(-1*r, tf.float16)


################################################################################
def load_training_dat_into_RAM(noise_folder, noNoise_folder):
	train, target = import_synth_data(noise_folder, noNoise_folder, BOX_DIM, NUM_NOISE_PAIRS)
	train = np.asarray(train, dtype='float16'); target = np.asarray(target,dtype='float16')

	#add extra dimension at end because only one color channel
	train = np.expand_dims(train, axis=-1)
	target = np.expand_dims(target, axis=-1)

	FRAC_VAL = int(train.shape[0] * TV_SPLIT)
	val_train = train[:FRAC_VAL]
	val_target = target[:FRAC_VAL]
	train = train[FRAC_VAL:]
	target = target[FRAC_VAL:]
	return train, target, val_train, val_target

print('Beginning training')
################################################################################
######### The data should be imported; now build the model #####################
################################################################################
# Define the model
def create_model_dense(training_data, full_training, lr):
	# Instantiate the model
	#input_img = layers.Input(shape=(training_data.shape[1:]))
	input_img = layers.Input(shape=(training_data.shape[1:]))
	# Make layers
	x = layers.Conv3D(64, kernel_size=(3,3,3), padding='same', activation='relu',trainable=full_training)(input_img) #[192x192x10]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x192 = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	
	x = layers.MaxPooling3D(pool_size=(4,4,4), padding='same')(x192)#[96x96x16]
	x = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv3D(168, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x96 = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	
	x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x96)#[48x48x64]
	x = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv3D(192, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x48 = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	
	x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x48)#[24x24x128]
	x = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[24x48x128]
	x = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[24x48x128]
	x = layers.Conv3D(256, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[24x48x128]
	x = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[24x48x128]
	x = layers.Dropout(0.20)(x)

	x = layers.UpSampling3D((2,2,2))(x)#[48x48x128]
	x = layers.Concatenate(axis=-1)([x, x48])#[48x48x256]
	x = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv3D(192, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	x = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[48x48x128]
	
	x = layers.UpSampling3D((2,2,2))(x)#[96x96x128]
	x = layers.Concatenate(axis=-1)([x, x96])#[192x192x192]
	x = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv3D(168, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	x = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[96x96x64]
	
	x = layers.UpSampling3D((4,4,4))(x)#[192x192x64]
	x = layers.Concatenate(axis=-1)([x, x192])#[192x192x80]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same')(x)
	decoded = layers.Conv3D(1, (1,1,1), activation='linear', padding='same',trainable=full_training)(x)#40
	
	# optimizer
	adam = keras.optimizers.Adam(learning_rate=lr)
	# Compile model
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer=adam, loss=CCC, metrics=['mse'])
	#autoencoder.compile(optimizer=adam, loss='mse', metrics=[CCC])
	autoencoder.summary()
	return autoencoder, x

################################################################################
# Handle model
def train_model(train_data, train_target, val_train, val_target):
	autoencoder, encoder = create_model_dense(train_data,True, LEARNING_RATE)
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=PATIENCE, restore_best_weights=True)
	checkpoint = keras.callbacks.ModelCheckpoint(OUTPUT_DIR_PATH +'fascinTomo_tester_CCC_{loss:.4f}_epoch_{epoch:02d}.h5', monitor='loss', verbose=0, save_best_only=True, mode='auto',save_weights_only=False, save_freq='epoch')
	history = autoencoder.fit(x=train_data, y=train_target, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es,checkpoint])
	return [autoencoder, history, encoder]

def train_model_generator(noise_list, noNoise_list):
	train_num = int(NUM_NOISE_PAIRS * (1-TV_SPLIT))
	autoencoder, encoder = create_model_dense(np.empty([1,BOX_DIM,BOX_DIM, BOX_DIM,1]),True, LEARNING_RATE)
	print('Training idxs = 0 to ' + str(train_num))
	print('Validation idxs = ' + str(train_num+1) + ' to ' + str(NUM_NOISE_PAIRS))
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=PATIENCE, restore_best_weights=True)
	checkpoint = keras.callbacks.ModelCheckpoint(OUTPUT_DIR_PATH +'fascinTomo_tester_CCC_{loss:.4f}_epoch_{epoch:02d}.h5', monitor='loss', verbose=0, save_best_only=True, mode='auto',save_weights_only=False, save_freq='epoch')
	train_gen = mrc_generator(noise_list, noNoise_list, box_length=BOX_DIM, start_idx=0, end_idx=train_num, batch_size=BATCH_SIZE)
	val_gen = mrc_generator(noise_list, noNoise_list, box_length=BOX_DIM, start_idx=train_num+1, end_idx=NUM_NOISE_PAIRS, batch_size=BATCH_SIZE)
	history = autoencoder.fit(train_gen, epochs=EPOCHS, verbose=1, steps_per_epoch = train_num // BATCH_SIZE, validation_data = val_gen, validation_steps= (NUM_NOISE_PAIRS-train_num+1) // BATCH_SIZE, callbacks=[es,checkpoint])
	return [autoencoder, history, encoder]


def continue_training(train_data, train_target, val_train, val_target, epochs, autoencoder_model):
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2, restore_best_weights=True)
	history = autoencoder_model.fit(x=train_data, y=train_target, epochs=epochs, batch_size=BATCH_SIZE, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es])
	return [autoencoder_model, history]

################################################################################
# Location of data
noise_folder = NOISY_DIR_PATH#folder + 'tplastin_noise/'#'noise_proj4/'
noNoise_folder = NOISELESS_DIR_PATH#folder + 'tplastin_noNoise/'#'noNoise_proj4/'

# Preload data into RAM or load training/validation data on the fly?
if(PRELOAD_RAM):
	train, target, val_train, val_target = load_training_dat_into_RAM(noise_folder, noNoise_folder)
	print('All files loaded and parsed into training and validation sets.')
	autoencoder_three, history_three, encoder_three = train_model(train,target, val_train, val_target)
else:
	noise_file_names = sorted(glob.glob(noise_folder+'*.mrc'))
	noNoise_file_names = sorted(glob.glob(noNoise_folder+'*.mrc'))
	autoencoder_three, history_three, encoder_three = train_model_generator(noise_file_names, noNoise_file_names)

# continue training, if needed
#autoencoder_longer_train, history_longer_train = continue_training(train[:100000], target[:100000], val_train, val_target, 2, autoencoder_three)

# save the final model
best_validation_loss = np.min(history_three.history['val_loss'])
print('After finishing training, the best validation loss was: ' + str(best_validation_loss))
model_save_name = OUTPUT_DIR_PATH + 'fascin_tomo_tester_box64_CCC%s.h5'%str(''.join(str(np.abs(best_validation_loss))[:6].replace('.','p')))
print('Model finished training.\nSaving model as ' + model_save_name)
autoencoder_three.save(model_save_name)

import pickle
with open(OUTPUT_DIR_PATH + 'myoXI_DAE_trainHistoryDict_fig.pkl', 'wb') as file_pi:
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
