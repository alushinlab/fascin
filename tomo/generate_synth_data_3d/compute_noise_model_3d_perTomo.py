#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import mrcfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
print('Imports finished. Beginning script...')
################################################################################
def pw_spectra(stack):
	pw = []
	for i in tqdm(range(0, len(stack))):
		temp = np.fft.fftn(stack[i])
		pw.append(np.fft.fftshift(temp))
		#pw.append(np.fft.fftshift(np.fft.fftn(stack[i])))

	return np.asarray(pw)

def compute_avg_spectrum_per_micrograph(file_name, NUM_RANDOM_SAMPLES):
    tomo = []
    print('Opening tomo: %s'%file_name)
    output_ID = file_name.split('/')[-1][:5]
    print(output_ID)
    with mrcfile.open(file_name, 'r') as mrc:
        tomo = mrc.data
    
    zdim, ydim, xdim = tomo.shape[0], tomo.shape[1], tomo.shape[2]
    # Load in 1000 random samples
    randsX = np.random.randint(int(xdim*0.25),int(xdim*0.75),NUM_RANDOM_SAMPLES)
    randsY = np.random.randint(int(ydim*0.25),int(ydim*0.75),NUM_RANDOM_SAMPLES)
    randsZ = np.random.randint(64,zdim-64,NUM_RANDOM_SAMPLES) # np.random.randint(int(zdim*0.25),int(zdim*0.75),NUM_RANDOM_SAMPLES)

    crop_holder = []
    print('Cropping micrograph sections...')
    for i in tqdm(range(0, NUM_RANDOM_SAMPLES)):
        box = tomo[randsZ[i]-half_box_size:randsZ[i]+half_box_size, 
            randsY[i]-half_box_size:randsY[i]+half_box_size, 
            randsX[i]-half_box_size:randsX[i]+half_box_size]
        #box = np.rot90(box, k=np.random.randint(0,3), axes=(1,2))
        crop_holder.append(box)

    crop_holder = np.asarray(crop_holder)

    print('Finished cropping out micrograph boxes. Computing power spectra...')
    pw = pw_spectra(crop_holder)

    print('Calculating average power spectrum of real data...')
    amp_mult = np.average(np.abs(pw),axis=0)
    amp_mult[half_box_size,half_box_size,half_box_size] = 0
    amp_mult_std = np.std(np.abs(pw),axis=0)
    amp_mult_std[half_box_size,half_box_size,half_box_size] = 0

    print('Saving file %s:...'%str(output_ID))
    with mrcfile.new('/rugpfs/fs0/cem/store/mreynolds/squiggle_cell_tomos/generate_synth_data_3d/empirical_noise_models_10p1Apix_box128_cryoCare/empirical_pink_noise_model_tomosAVG_%s.mrc'%str(output_ID), overwrite=True) as mrc:
        mrc.set_data(np.asarray(amp_mult).astype('float32'))

    with mrcfile.new('/rugpfs/fs0/cem/store/mreynolds/squiggle_cell_tomos/generate_synth_data_3d/empirical_noise_models_10p1Apix_box128_cryoCare/empirical_pink_noise_model_tomosSTD_%s.mrc'%str(output_ID), overwrite=True) as mrc:
        mrc.set_data(np.asarray(amp_mult_std).astype('float32'))


################################################################################
file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/squiggle_cell_tomos/cryoCare_tomos_bin2/*.mrc'))
box_size = 128
half_box_size = int(box_size / 2)

for i in range(0, len(file_names)):
    if('503' not in file_names[i]):
        compute_avg_spectrum_per_micrograph(file_names[i], 50)


