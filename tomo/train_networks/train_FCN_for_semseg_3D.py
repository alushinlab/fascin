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
from scipy import interpolate; from scipy.ndimage import filters
import glob
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
parser.add_argument('--preload_ram', type=str, help='Pre-load all training data to RAM? If yes, type True')
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

def mrc_generator(noisy_file_list, semMap_file_list, box_length, start_idx, end_idx, batch_size=32):
	while True:
		# Loop over batches of files
		for i in range(start_idx, end_idx, batch_size):
			# Load the batch of files
			batch_files_noisy = noisy_file_list[i:i+batch_size]
			batch_files_semMap_actin = semMap_file_list[0][i:i+batch_size]
			batch_files_semMap_fascin = semMap_file_list[1][i:i+batch_size]
			batch_files_semMap_background = semMap_file_list[2][i:i+batch_size]

			# Initialize the input and output arrays
			batch_x = np.zeros((len(batch_files_noisy), box_length, box_length, box_length, 1), dtype=np.float16)
			batch_y = np.zeros((len(batch_files_semMap_actin), box_length, box_length, box_length, 3), dtype=np.float16)
			
			# Loop over the files in the batch
			for j in range(0, len(batch_files_noisy)):
				# Read the MRC volume file
				with mrcfile.open(batch_files_noisy[j], mode='r', permissive=True) as mrc:
					noisy_data = mrc.data.astype(np.float16)
				with mrcfile.open(batch_files_semMap_actin[j], mode='r', permissive=True) as mrc:
					noiseless_data_actin = mrc.data.astype(np.float16)
				
				with mrcfile.open(batch_files_semMap_fascin[j], mode='r', permissive=True) as mrc:
					noiseless_data_fascin = mrc.data.astype(np.float16)
				
				with mrcfile.open(batch_files_semMap_background[j], mode='r', permissive=True) as mrc:
					noiseless_data_background = mrc.data.astype(np.float16)
				

				if(np.isnan(noisy_data).any() or np.isnan(noiseless_data_actin).any() or np.isnan(noiseless_data_fascin).any() or np.isnan(noiseless_data_background).any()):
					continue
				
				if(noisy_data.shape == (box_length,box_length,box_length) and noiseless_data_actin.shape == (box_length,box_length,box_length) and noiseless_data_fascin.shape == (box_length,box_length,box_length) and noiseless_data_background.shape == (box_length,box_length,box_length)):
					# Add the volume to the input array
					batch_x[j, :, :, :, 0] = noisy_data
					batch_y[j, :, :, :, 0] = noiseless_data_actin
					batch_y[j, :, :, :, 1] = noiseless_data_fascin
					batch_y[j, :, :, :, 2] = noiseless_data_background
			
			yield (batch_x, batch_y)



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

################################################################################
def create_model_dense(training_data, full_training, lr):
	#input_img = layers.Input(shape=(training_data.shape[1:]))
	input_img = layers.Input(shape=(np.empty([1,BOX_DIM,BOX_DIM, BOX_DIM,1]).shape[1:]))
	x = layers.Conv3D(64, kernel_size=(3,3,3), padding='same', activation='relu',trainable=False)(input_img) #[192x192x10]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(1,1,1), activation='relu', padding='same',trainable=False)(x)#[192x192x16]
	x192 = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[192x192x16]
	
	x = layers.MaxPooling3D(pool_size=(4,4,4), padding='same')(x192)#[96x96x16]
	x = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[96x96x64]
	x = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[96x96x64]
	x = layers.Conv3D(168, kernel_size=(1,1,1), activation='relu', padding='same',trainable=False)(x)#[96x96x64]
	x96 = layers.Conv3D(168, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[96x96x64]
	
	x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x96)#[48x48x64]
	x = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[48x48x128]
	x = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[48x48x128]
	x = layers.Conv3D(192, kernel_size=(1,1,1), activation='relu', padding='same',trainable=False)(x)#[48x48x128]
	x48 = layers.Conv3D(192, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[48x48x128]
	
	x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x48)#[24x24x128]
	x = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[24x48x128]
	x = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[24x48x128]
	x = layers.Conv3D(256, kernel_size=(1,1,1), activation='relu', padding='same',trainable=False)(x)#[24x48x128]
	x = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same',trainable=False)(x)#[24x48x128]
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
	decoded = layers.Conv3D(3, (1,1,1), activation='softmax', padding='same',trainable=True)(x)#40
	
	# optimizer
	adam = keras.optimizers.Adam(lr=lr)
	# Compile model
	semSeg = Model(input_img, decoded)
	#semSeg.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
	semSeg.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
	semSeg.summary()
	return semSeg

def create_shared_layers(full_training):
	input_img = layers.Input(shape=(np.empty([1,BOX_DIM,BOX_DIM, BOX_DIM,1]).shape[1:]))
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
	return Model(inputs=input_img, outputs=x)

def create_model_semSeg(shared_layers, full_training):
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(shared_layers)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(shared_layers)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(shared_layers)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(shared_layers)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)
	# Decoded produces the final image
	decoded = layers.Conv3D(3, (1,1,1), activation='softmax', padding='same',trainable=full_training)(x)#40
	return decoded


################################################################################
############################### Import the data ################################
################################################################################
# Import the encoding layers of the DAE model
model_path = TRAINED_DAE_PATH#'../300000training_tplastin_CCC09856.h5'
trained_DAE = keras.models.load_model(model_path, custom_objects={'CCC':CCC})
trained_DAE.summary()
# Location of data
noise_folder = NOISY_DIR_PATH#folder + 'tplastin_noise/'#'noise_proj4/'
noNoise_folder = SEMSEG_DIR_PATH#folder + 'tplastin_noNoise/'#'noNoise_proj4/'

# Preload data into RAM or load training/validation data on the fly?
# Define the model
shared_layers = create_shared_layers(False)
semSeg_input = layers.Input(shape=(np.empty([1,BOX_DIM,BOX_DIM, BOX_DIM,1]).shape[1:]))
shared_layers_dae = shared_layers(semSeg_input)
semSeg_output = create_model_semSeg(shared_layers_dae, True)
semSeg_model = Model(inputs=semSeg_input, outputs=semSeg_output)
adam_semSeg = keras.optimizers.Adam(learning_rate=0.0001)
semSeg_model.compile(optimizer=adam_semSeg, loss='categorical_crossentropy', metrics=['categorical_crossentropy'])

shared_layers.summary()
semSeg_model.summary()
for i in range(0, len(trained_DAE.layers)-1): # set to -5 for the actual discriminator
	semSeg_model.layers[i].set_weights(trained_DAE.layers[i-6].get_weights())



#semSeg = create_model_dense(np.empty([1,BOX_DIM,BOX_DIM,BOX_DIM,1]),True, LEARNING_RATE) #0.00005
#for i in range(0, len(trained_DAE.layers)-1):
#	semSeg.layers[i].set_weights(trained_DAE.layers[i].get_weights())

#for i in range(2, len(trained_DAE.layers)): # set to -5 for the actual discriminator
#	semSeg.layers[i].set_weights(trained_DAE.layers[i-6].get_weights())


# Train the model. If not loading data on the fly, preload data first
if(PRELOAD_RAM):
	train, target, val_train, val_target = load_training_dat_into_RAM(noise_folder, noNoise_folder)
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=PATIENCE, restore_best_weights=True, save_freq='epoch')
	history = semSeg.fit(x=train, y=target, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data = (val_train[1:], val_target[1:]), callbacks=[es])
	print('All files loaded and parsed into training and validation sets.')
	autoencoder_three, history_three, encoder_three = train_model(train,target, val_train, val_target)
else:
	noise_file_names = sorted(glob.glob(noise_folder+'*.mrc'))
	semMap_file_names_actin = sorted(glob.glob(noNoise_folder+'actin*.mrc'))
	semMap_file_names_fascin = sorted(glob.glob(noNoise_folder+'fascin*.mrc'))
	semMap_file_names_background = sorted(glob.glob(noNoise_folder+'background*.mrc'))
	semMap_file_names = [semMap_file_names_actin, semMap_file_names_fascin, semMap_file_names_background]
	train_num = int(NUM_NOISE_PAIRS * (1-TV_SPLIT))
	print('Training idxs = 0 to ' + str(train_num))
	print('Validation idxs = ' + str(train_num+1) + ' to ' + str(NUM_NOISE_PAIRS))
	es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=PATIENCE, restore_best_weights=True)
	checkpoint = keras.callbacks.ModelCheckpoint(OUTPUT_DIR_PATH +'fascin_CCE_{loss:.4f}_epoch_{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min',save_weights_only=False, save_freq='epoch')
	train_gen = mrc_generator(noise_file_names, semMap_file_names, box_length=BOX_DIM, start_idx=0, end_idx=train_num, batch_size=BATCH_SIZE)
	val_gen = mrc_generator(noise_file_names, semMap_file_names, box_length=BOX_DIM, start_idx=train_num+1, end_idx=NUM_NOISE_PAIRS, batch_size=BATCH_SIZE)
	history = semSeg_model.fit(train_gen, epochs=EPOCHS, verbose=1, steps_per_epoch = train_num // BATCH_SIZE, validation_data = val_gen, validation_steps= (NUM_NOISE_PAIRS-train_num+1) // BATCH_SIZE, callbacks=[es,checkpoint])
	#return [semSeg, history, encoder]

################################################################################

model_save_name = OUTPUT_DIR_PATH + 'semSeg_50ktrain_catCrossEnt.h5'
print('Model finished training.\nSaving model as ' + model_save_name)
semSeg_model.save(model_save_name)
print('Model saved.')



import pickle
with open(OUTPUT_DIR_PATH + 'semSeg_trainHistoryDict.pkl', 'wb') as file_pi:
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




