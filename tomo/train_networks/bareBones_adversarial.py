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
from keras.layers import Layer
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
def random_rotation(image_array: np.ndarray):
    # pick a random number of rotations
    k = random.choice([1, 2, 3])
    # pick a random plane to rotate about
    axes = random.choice([(0, 1), (0, 2), (1, 2)])
    return np.rot90(image_array, k, axes)

# method to import synthetic data from files
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
				
				#noisy_data = random_rotation(noisy_data)
				#noiseless_data = random_rotation(noiseless_data)
				
				if(noisy_data.shape == (box_length,box_length,box_length) and noiseless_data.shape == (box_length,box_length,box_length)):
					# Add the volume to the input array
					batch_x[j, :, :, :, 0] = noisy_data
					batch_y[j, :, :, :, 0] = noiseless_data
			
			yield (batch_x, batch_y)

def adversary_generator(adv_mrc_list, adv_label_list, box_length, start_idx, end_idx, batch_size=32):
	while True:
		# Loop over batches of files
		for i in range(start_idx, end_idx, batch_size):
			# Load the batch of files
			batch_files_mrc = adv_mrc_list[i:i+batch_size]
			batch_files_labels = adv_label_list[i:i+batch_size]

			# Initialize the input and output arrays
			batch_x = np.zeros((len(batch_files_mrc), box_length, box_length, box_length, 1), dtype=np.float16)
			batch_y = tf.cast(np.expand_dims(batch_files_labels, axis=-1), dtype=np.float16)
			
			# Loop over the files in the batch
			for j in range(0, len(batch_files_mrc)):
				# Read the MRC volume file
				with mrcfile.open(batch_files_mrc[j], mode='r', permissive=True) as mrc:
						noisy_data = mrc.data.astype(np.float16)

				#noisy_data = random_rotation(noisy_data)
				if(np.isnan(noisy_data).any()):
					batch_x[j, :, :, :, 0] = np.zeros_like(noisy_data)
				elif(noisy_data.shape == (box_length,box_length,box_length)):
					# Add the volume to the input array
					batch_x[j, :, :, :, 0] = noisy_data
			
			yield (batch_x, batch_y)


################################################################################
#https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
import keras.backend as K
def CCC(y_pred, y_true):
	x = y_true
	y = y_pred
	mx=K.mean(x)
	my=K.mean(y)
	xm, ym = x-mx, y-my
	r_num = K.sum(tf.multiply(xm,ym))
	r_den = K.sqrt(tf.maximum(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))), K.epsilon()))
	r = r_num / r_den
	#return -1*r
	if tf.math.is_nan(r):
		return tf.cast(1, tf.float16)
	else:
		return tf.cast(-1*r, tf.float16)

@tf.custom_gradient
def grad_reverse(x):
	y = tf.identity(x)
	def custom_grad(dy):
		return -dy
	return y, custom_grad

class GradientReversalLayer(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)

def set_trainable_dae(model, value):
    for layer in model.layers:
        layer.trainable = value
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss=CCC, metrics=['mse'])

def set_trainable_disc(model, value):
    for layer in model.layers:
        layer.trainable = value
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0000001), loss='BCE', metrics=['accuracy'])


################################################################################
print('Beginning training')
################################################################################
######### The data should be imported; now build the model #####################
################################################################################
# Define the model
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

def create_model_dae(shared_layers, full_training):
	x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(shared_layers)#[192x192x16]
	x = layers.Conv3D(128, kernel_size=(1,1,1), activation='relu', padding='same',trainable=full_training)(x)#[192x192x16]
	x = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same',trainable=full_training)(x)
	# Decoded produces the final image
	decoded = layers.Conv3D(1, (1,1,1), activation='linear', padding='same',trainable=full_training)(x)#40
	return decoded

def create_model_discriminator(shared_layers, adv_training):
	# Define the descriminator network
	grl_layer = GradientReversalLayer()(shared_layers)
	shared_layers = layers.Concatenate()([shared_layers, grl_layer])
	disc = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same',trainable=adv_training)(shared_layers)
	disc = layers.MaxPooling3D(pool_size=(4,4,4), padding='same')(disc)#[24x24x128]
	disc = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same',trainable=adv_training)(disc)
	disc = layers.MaxPooling3D(pool_size=(4,4,4), padding='same')(disc)#[24x24x128]
	disc = layers.Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same',trainable=adv_training)(disc)
	disc = layers.Flatten()(disc)
	disc = layers.Dense(128, activation='sigmoid',trainable=adv_training)(disc)
	disc = layers.Dense(1, activation='sigmoid',trainable=adv_training)(disc)
	return disc


################################################################################
################################################################################
# Location of purley synthetic data
noise_folder = NOISY_DIR_PATH#folder + 'tplastin_noise/'#'noise_proj4/'
noNoise_folder = NOISELESS_DIR_PATH#folder + 'tplastin_noNoise/'#'noNoise_proj4/'

# Preload data into RAM or load training/validation data on the fly?
noise_file_names = sorted(glob.glob(noise_folder+'*.mrc'))
noNoise_file_names = sorted(glob.glob(noNoise_folder+'*.mrc'))

# Location of real data for domain target transfer
real_data_folder = # PATH TO REAL DATA
real_data_file_names_adv = sorted(glob.glob(real_data_folder+'*707*.mrc'))[:500]
synth_data_file_names_adv = sorted(glob.glob(noise_folder+'*.mrc'))[:len(real_data_file_names_adv)]
adv_labels = np.concatenate((np.ones(len(real_data_file_names_adv)),np.zeros(len(synth_data_file_names_adv))))
adv_file_names = np.concatenate((real_data_file_names_adv,synth_data_file_names_adv))
combined_adv_file_names = list(zip(adv_file_names, adv_labels))
random.shuffle(combined_adv_file_names)
shuffled_adv_file_names, shuffled_adv_labels = zip(*combined_adv_file_names)

print('Passed data import code block')
###############################################################################
############################# Create and Train models #########################
###############################################################################
shared_layers = create_shared_layers(True)
dae_input = layers.Input(shape=(np.empty([1,BOX_DIM,BOX_DIM, BOX_DIM,1]).shape[1:]))
disc_input = layers.Input(shape=(np.empty([1,BOX_DIM,BOX_DIM, BOX_DIM,1]).shape[1:]))
shared_layers_dae = shared_layers(dae_input)
shared_layers_disc = shared_layers(disc_input)
dae_output = create_model_dae(shared_layers_dae, True)
disc_output = create_model_discriminator(shared_layers_disc, False)


dae_model = Model(inputs=dae_input, outputs=dae_output)
disc_model = Model(inputs=disc_input, outputs=disc_output)


adam_dae = keras.optimizers.Adam(learning_rate=0.0000001)
adam_disc = keras.optimizers.Adam(learning_rate=0.000000001)
dae_model.compile(optimizer=adam_dae, loss=CCC, metrics=['mse'])
disc_model.compile(optimizer=adam_disc, loss='BCE', metrics=['accuracy'])

shared_layers.summary()
dae_model.summary()
disc_model.summary()

pretrained_DAE_path = # PATH TO PRETRAINED DAE
pretrained_DAE = keras.models.load_model(pretrained_DAE_path, custom_objects={'CCC':CCC})

for i in range(0, len(shared_layers.layers)): # set to -5 for the actual discriminator
	shared_layers.layers[i].set_weights(pretrained_DAE.layers[i].get_weights())

for i in range(2, len(dae_model.layers)): # set to -5 for the actual discriminator
	dae_model.layers[i].set_weights(pretrained_DAE.layers[i-6].get_weights())


train_num = int(NUM_NOISE_PAIRS * (1-TV_SPLIT))
train_gen_dae = mrc_generator(noise_file_names, noNoise_file_names, box_length=BOX_DIM, start_idx=0, end_idx=train_num, batch_size=BATCH_SIZE)
val_gen_dae = mrc_generator(noise_file_names, noNoise_file_names, box_length=BOX_DIM, start_idx=train_num+1, end_idx=NUM_NOISE_PAIRS, batch_size=BATCH_SIZE)
train_gen_adv = adversary_generator(shuffled_adv_file_names, shuffled_adv_labels, box_length=BOX_DIM, start_idx=0, end_idx=train_num, batch_size=BATCH_SIZE)#adversary_generator(shuffled_adv_file_names, shuffled_adv_labels, box_length=BOX_DIM, start_idx=0, end_idx=train_num, batch_size=BATCH_SIZE)
val_gen_adv = adversary_generator(shuffled_adv_file_names, shuffled_adv_labels, box_length=BOX_DIM, start_idx=train_num+1, end_idx=NUM_NOISE_PAIRS, batch_size=BATCH_SIZE)#adversary_generator(shuffled_adv_file_names, shuffled_adv_labels, box_length=BOX_DIM, start_idx=train_num+1, end_idx=NUM_NOISE_PAIRS, batch_size=BATCH_SIZE)

# Do the training
n_epochs = 10
steps_per_epoch = int(np.ceil(train_num / BATCH_SIZE))  # Assuming train_num is your total number of training samples
for epoch in range(n_epochs):
    print(f"Start of epoch {epoch}")
    set_trainable_dae(dae_model, True)  # unfreeze the DAE model
    set_trainable_disc(disc_model, False)  # freeze the discriminator
    for step in range(steps_per_epoch):
        dae_batch_X, dae_batch_Y = next(train_gen_dae)
        dae_train_loss = dae_model.train_on_batch(dae_batch_X, dae_batch_Y)
        print(f"Step: {step}, DAE Training Loss (CCC): {dae_train_loss[0]}, Metric MSE: {dae_train_loss[1]}")

    set_trainable_dae(dae_model, False)  # freeze the DAE model
    set_trainable_disc(disc_model, True)  # unfreeze the discriminator
    for step in range(steps_per_epoch):
        disc_batch_X, disc_batch_Y = next(train_gen_adv)
        disc_train_loss = disc_model.train_on_batch(disc_batch_X, disc_batch_Y)
        if step % 1 == 0:  # print every 100 steps
            print(f"Step: {step}, Discriminator Training Loss (BCE): {disc_train_loss[0]}, Metric Accuracy: {disc_train_loss[1]}")

    # Evaluation at the end of the epoch
    dae_val_loss = dae_model.evaluate(val_gen_dae, steps=np.ceil((NUM_NOISE_PAIRS - train_num - 1) / BATCH_SIZE), verbose=0)
    disc_val_loss = disc_model.evaluate(val_gen_adv, steps=np.ceil((NUM_NOISE_PAIRS - train_num - 1) / BATCH_SIZE), verbose=0)
    print(f"End of epoch {epoch}, DAE Validation Loss (CCC): {dae_val_loss[0]}, DAE Validation Metric (MSE): {dae_val_loss[1]}, Discriminator Validation Loss (BCE): {disc_val_loss[0]}, Discriminator Validation Metric (accuracy): {disc_val_loss[1]}")
    dae_model.save(OUTPUT_DIR_PATH + 'adversarial_trained_DAE_epoch_' +str(epoch).zfill(3)+'.h5')
    disc_model.save(OUTPUT_DIR_PATH + 'adversarial_trained_discriminator_epoch_' +str(epoch).zfill(3)+'.h5')

