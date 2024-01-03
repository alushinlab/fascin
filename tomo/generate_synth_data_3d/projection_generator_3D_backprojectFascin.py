#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Generate noisy and noiseless 2D projections of randomly oriented and translated MRC files.')
# IO parameters
parser.add_argument('--input_mrc_dir', type=str, help='Input directory containing MRC files to be rotated, translated, CTF-convolved, and projected')
parser.add_argument('--output_noise_dir', type=str, help='Output directory to store noisy 2D projections')
parser.add_argument('--output_noiseless_dir', type=str, help='Output directory to store noiseless 2D projections')
parser.add_argument('--output_semMap_dir', type=str, help='Output directory to store semantic segmentation targets')
parser.add_argument('--numProjs', type=int, help='Total number of projections to make')
parser.add_argument('--nProcs', type=int, help='Total number of parallel threads to launch')
parser.add_argument('--box_len', type=int, help='Box size of projected image')
# Is filament?
#parser.add_argument('--is_filament', type=int, help='Boolean. Are you modelling a filament? If so, use more detailed angular samplings')
# Bundle parameters
parser.add_argument('--alt_stdev', type=float, help='Floating point. Standard deviation of the randomly generated altitude values (tilt in RELION), sampled from Gaussian distribution centered at 0.')
parser.add_argument('--tx_low', type=int, help='Integer. Minimum translation of object in x (to left), in pixels.')
parser.add_argument('--tx_high', type=int, help='Integer. Maximum translation of object in x (to right), in pixels.')
parser.add_argument('--ty_low', type=int, help='Integer. Minimum translation of object in y (down), in pixels.')
parser.add_argument('--ty_high', type=int, help='Integer. Maximum translation of object in y (up), in pixels.')
parser.add_argument('--tz_low', type=int, help='Integer. Minimum translation of object in z (out of screen), in pixels.')
parser.add_argument('--tz_high', type=int, help='Integer. Maximum translation of object in z (into screen), in pixels.')
parser.add_argument('--max_fil_num', type=int, help='Integer. Maximum number of filament units in an image. Each produced image will have at minimum zero and at maximum this many filaments in it, uniformly, discretely sampled.')
parser.add_argument('--is_bundle', type=str, help='Boolean. Do you wish to simulate bundled filaments?')
parser.add_argument('--bundle_freq', type=float, help='Floating point. Frequency of each image being a bundle. 0=Never a bundle; 1=Always a bundle.')
parser.add_argument('--bundle_alt_stdev', type=float, help='Floating point. Standard deviation of the randomly generated altitude values (tilt in RELION), sampled from a Gaussian distribution, relative to each other. 0 Means filaments are always perfectly parallel. ')
parser.add_argument('--bundle_type', type=str, help='What polarity orientation do these filaments have? If both, 50 percent will be parallel, 50 percent will be anti-parallel. Enter either \"Parallel\", \"Antiparallel\", or \"Both\"; case-sensitive.')
parser.add_argument('--bundle_dist', type=float, help='Floating point. Minimum translation of filament 2 in x (left), relative to filament 1, in pixels.')
parser.add_argument('--bundle_dist_stdev', type=float, help='Floating point. Minimum translation of filament 2 in x (down), relative to filament 1, in pixels.')
parser.add_argument('--bundle_ty_low', type=int, help='Integer. Minimum translation of filament 2 in y (down), relative to filament 1, in pixels.')
parser.add_argument('--bundle_ty_high', type=int, help='Integer. Maximum translation of filament 2 in y (up), relative to filament 1, in pixels.')
# Synthetic noise parameters
parser.add_argument('--kV', type=int, help='Voltage of the electron microscope in kV')
parser.add_argument('--ampCont', type=float, help='Amplitude contrast (from 0 to 100)')
parser.add_argument('--angpix', type=float, help='Angstroms per pixel')
parser.add_argument('--bfact', type=float, help='Bfactor')
parser.add_argument('--cs', type=float, help='Spherical Aberration')
parser.add_argument('--defoc_low', type=float, help='Lower end of defocus range (1.0 means 1 micron underfocused)')
parser.add_argument('--defoc_high', type=float, help='Upper end of defocus range (4.0 means 4 microns underfocused)')
parser.add_argument('--noise_amp', type=float, help='Average amplitude of pink noise curve')
parser.add_argument('--noise_stdev', type=float, help='Standard deviation of amplitudes of pink noise curves')
parser.add_argument('--use_empirical_model', type=str, help='Boolean. Are you using an empirical model? If so, specify the path with that option.')
parser.add_argument('--path_to_empirical_model', type=str, help='String. Relative or absolute path to the empirical model.')
parser.add_argument('--path_to_empirical_model_STD', type=str, help='String. Relative or absolute path to the empirical model.')
# Semantic segmentation parameters
parser.add_argument('--lowpass_res', type=float, help='Resolution to which binarized 2D projections are lowpass-filtered, in Angstroms.')
parser.add_argument('--dil_rad', type=float, help='Dilation radius of binarized, lowpass filtered projections, in pixels.')
parser.add_argument('--erod_rad', type=float, help='Erosion radius of binarized, lowpass filtered projections, in pixels.')

args = parser.parse_args()
print('')
if(args.input_mrc_dir == None or args.output_noise_dir == None or args.output_noiseless_dir == None):
	print('Please enter an input_mrc_dir, AND an output_noise_dir, AND an output_noiseless_dir')
	sys.exit('The preferred input style may be found with ./projection_generator.py -h')

if(args.numProjs == None):
	sys.exit('Please enter the number of projection images you would like with the --numProjs flag')

if(args.box_len == None or args.kV == None or args.angpix == None or args.bfact == None or args.cs == None
	or args.defoc_low == None or args.defoc_high == None or args.noise_amp == None or args.noise_stdev == None
	or args.lowpass_res == None or args.dil_rad == None or args.erod_rad == None):
	sys.exit('Please enter all noise model inputs.')

if(args.nProcs == None):
	print('No process number specified, using one thread')
	nProcs = 1
else:
	nProcs = int(args.nProcs)

if(args.numProjs % nProcs != 0):
	print('The numProjs that you specified was not a multiple of nProcs.')
	print('Instead of %d 2D projections, this program will generate %d 2D projections'%(args.numProjs, args.numProjs/nProcs*nProcs))

folder = args.input_mrc_dir
noNoise_outputDir = args.output_noiseless_dir
noise_outputDir = args.output_noise_dir
semMap_outputDir = args.output_semMap_dir
TOTAL_NUM_TO_MAKE = args.numProjs
#image sizes
cropBox = args.box_len; half_box = int(int(cropBox)/2)
center = cropBox

# Noise parameters
# microscope parameters
voltage = args.kV #300
ampCont = args.ampCont # 10.0
apix = args.angpix#4.12
bfact = args.bfact#0.0
cs = args.cs#2.7
defoc_low = args.defoc_low; defoc_high = args.defoc_high#1.0, 4.0
noise_amp = args.noise_amp; noise_stdev = args.noise_stdev # 0.050, 0.010
# empirical noise model inputs
use_empirical_model = (args.use_empirical_model == 'True' or args.use_empirical_model == 'true' or args.use_empirical_model == 'y' or args.use_empirical_model == '1' or args.use_empirical_model == 'yes')
path_to_empirical_model_AVG = args.path_to_empirical_model
path_to_empirical_model_STD = args.path_to_empirical_model_STD


# bundle parameters
alt_stdev = args.alt_stdev # 7.5
tx_low  = args.tx_low  #-60
tx_high = args.tx_high # 60 
ty_low  = args.ty_low  #-60
ty_high = args.ty_high # 60 
tz_low  = args.tz_low  #-60
tz_high = args.tz_high # 60
max_fil_num = args.max_fil_num # 3
is_bundle = (args.is_bundle == 'True' or args.is_bundle == 'true' or args.is_bundle == 'yes' or args.is_bundle == 'y' or args.is_bundle == '1')     # True
bundle_freq = args.bundle_freq # 0.65
bundle_alt_stdev  = args.bundle_alt_stdev  # 1.5
bundle_type = args.bundle_type # 'Parallel'
bundle_dist = args.bundle_dist # 113.8 A
bundle_dist_stdev = args.bundle_dist_stdev # 15.85 A
bundle_ty_low  = args.bundle_ty_low  # -44
bundle_ty_high = args.bundle_ty_high #  44
#if(is_bundle == '0'):
#	max_fil_num = 1

#print(args)


# processing parameters
lowpass_res = args.lowpass_res #40.0
dilation_radius = args.dil_rad #8
erosion_radius = args.erod_rad #16

if(folder[-1] != '/'): folder = folder + '/'
if(noNoise_outputDir[-1] != '/'): noNoise_outputDir = noNoise_outputDir + '/'
if(noise_outputDir[-1] != '/'): noise_outputDir = noise_outputDir + '/'
if(semMap_outputDir[-1] != '/'): semMap_outputDir = semMap_outputDir + '/'
print('The program will now generate %d 2D projections'%(args.numProjs/args.nProcs*args.nProcs))
################################################################################
# import of python packages
import numpy as np
from EMAN2 import *; from sparx import *; import mrcfile
import json; import glob
from multiprocessing import Pool
import os; from tqdm import tqdm
from skimage.morphology import erosion,dilation; from skimage.morphology import disk, ball
################################################################################
################################################################################
# import data
fascins_and_filaments = []
#file_names = sorted(os.listdir(folder))
filament_file_names = sorted(glob.glob(folder+'filaments*.mrc'))
fascin_file_names = sorted(glob.glob(folder+'fascin*.mrc'))
for i in range(0, len(filament_file_names)):
	temp = [EMData(filament_file_names[i]), EMData(fascin_file_names[i])]
	fascins_and_filaments.append(temp)

if(use_empirical_model):
	pwrSpectrum_paths_avg = sorted(glob.glob(path_to_empirical_model_AVG + '*AVG*.mrc'))
	pwrSpectrum_paths_std = sorted(glob.glob(path_to_empirical_model_STD + '*STD*.mrc'))
	empirical_pwrSpectrum = []
	empirical_pwrSpectrum_std = []
	for i in range(0, len(pwrSpectrum_paths_avg)):
		with mrcfile.open(pwrSpectrum_paths_avg[i], 'r') as mrc:
			mrc = mrc.data
			empirical_pwrSpectrum.append(mrc)

		with mrcfile.open(pwrSpectrum_paths_std[i], 'r') as mrc:
			mrc = mrc.data
			empirical_pwrSpectrum_std.append(mrc)

	empirical_pwrSpectrum = np.asarray(empirical_pwrSpectrum)
	empirical_pwrSpectrum_std = np.asarray(empirical_pwrSpectrum_std)

################################################################################
################################################################################
def launch_parallel_process(thread_idx):
	index=num_per_proc*thread_idx
	for i in tqdm(range(0,num_per_proc),file=sys.stdout):
		local_random_state = np.random.RandomState(None)
		# First: randomly pick one of the actin mrc files that were loaded into actin_orig
		r0 = local_random_state.randint(0,len(fascins_and_filaments))
		num_filaments = local_random_state.choice([0,max_fil_num],p=[0.2,0.8])
		r15 = local_random_state.randint(0,len(empirical_pwrSpectrum))
		if(num_filaments == 0):
			target = np.concatenate((np.zeros((2,cropBox,cropBox,cropBox)), np.ones((1,cropBox,cropBox,cropBox))), axis=0)
			r14 = local_random_state.normal(0,0.5)
			white_box = np.random.normal(0,0.4,[cropBox*2,cropBox*2,cropBox*2])
			pnk_pw = np.multiply(empirical_pwrSpectrum[r15]+r14*empirical_pwrSpectrum_std[r15], white_box)
			pnk_pw_shift = np.fft.ifftshift(pnk_pw)
			pink_noise_box = np.fft.ifftn(pnk_pw_shift).real
			pink_noise_box = ((pink_noise_box - np.average(pink_noise_box)) / np.std(pink_noise_box))
			target_noise = pink_noise_box.real
			target_noise = target_noise[cropBox-half_box:cropBox+half_box, cropBox-half_box:cropBox+half_box, cropBox-half_box:cropBox+half_box]
			target_noise = (target_noise - np.mean(target_noise)) / np.std(target_noise) # normalize
			with mrcfile.new(noise_outputDir + 'actin_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_noise.astype('float32'))
			
			with mrcfile.new(noNoise_outputDir + 'actin_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target[0].astype('float32'))
			
			
			semMap = np.concatenate((np.zeros((2,cropBox,cropBox,cropBox)), np.ones((1,cropBox,cropBox,cropBox))), axis=0)
			with mrcfile.new(semMap_outputDir + 'actin_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(semMap[0].astype('float32'))
			
			with mrcfile.new(semMap_outputDir + 'fascin_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(semMap[1].astype('float32'))
			
			with mrcfile.new(semMap_outputDir + 'background_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(semMap[2].astype('float32'))
			
		else:
			target = np.zeros((max_fil_num+2,cropBox,cropBox,cropBox)); target_lp50 = np.zeros((max_fil_num+2,cropBox,cropBox,cropBox))
			for j in range(1, num_filaments+1):
				#r0_name = file_names[r0]
				rotated_actin = fascins_and_filaments[r0][0].copy()
				rotated_fascin = fascins_and_filaments[r0][1].copy()

				# Rotation angles: azimuth, alt, phi, then Translations: tx, ty,tz
				r1, r2, r3 = int(local_random_state.random_sample()*360),int(local_random_state.normal(loc=90, scale=alt_stdev)),int(local_random_state.random_sample()*360)
				r4, r5, r6 = local_random_state.uniform(tx_low, tx_high), local_random_state.uniform(ty_low, ty_high), local_random_state.uniform(tz_low, tz_high)
				t = Transform()
				t.set_params({'type':'eman','az':r1, 'alt':r2, 'phi':r3, 'tx':r4, 'ty':r5, 'tz':r6})
				rotated_actin.transform(t) # apply rotation and translation
				rotated_actin_lp = rotated_actin.copy().process('filter.lowpass.gauss', {'apix':apix, 'cutoff_freq':1.0/lowpass_res})
				rotated_actin_lp_np = EMNumPy.em2numpy(rotated_actin_lp)
				rotated_actin_lp_np = rotated_actin_lp_np>0.1
				#rotated_actin_lp_np = dilation(rotated_actin_lp_np>0.1, footprint=ball(dilation_radius))
				#rotated_actin_lp_np = erosion(rotated_actin_lp_np, footprint=ball(erosion_radius))
				rotated_actin_np = EMNumPy.em2numpy(rotated_actin)
				
				rotated_fascin.transform(t) # apply rotation and translation
				rotated_fascin_lp = rotated_fascin.copy().process('filter.lowpass.gauss', {'apix':apix, 'cutoff_freq':1.0/lowpass_res})
				rotated_fascin_lp_np = EMNumPy.em2numpy(rotated_fascin_lp)
				rotated_fascin_lp_np = rotated_fascin_lp_np>0.1
				#rotated_fascin_lp_np = dilation(rotated_fascin_lp_np>0.1, footprint=ball(dilation_radius))
				#otated_fascin_lp_np = erosion(rotated_fascin_lp_np, footprint=ball(erosion_radius))
				rotated_fascin_np = EMNumPy.em2numpy(rotated_fascin)

				

				center = cropBox
				target_filament = rotated_actin_np[center-half_box:center+half_box, center-half_box:center+half_box, center-half_box:center+half_box]
				target_filament = target_filament + rotated_fascin_np[center-half_box:center+half_box, center-half_box:center+half_box, center-half_box:center+half_box]
				target_filament = (target_filament - np.mean(target_filament)) / np.std(target_filament) # normalize
				target[j] = target_filament
				
				target_filament_lp50 = rotated_actin_lp_np[center-half_box:center+half_box, center-half_box:center+half_box,center-half_box:center+half_box]
				if(np.std(target_filament_lp50 > 0.0001)):
					target_filament_lp50 = (target_filament_lp50 - np.mean(target_filament_lp50)) / np.std(target_filament_lp50) # normalize
				target_lp50[j] = target_filament_lp50
				
				target_filament_lp50 = rotated_fascin_lp_np[center-half_box:center+half_box, center-half_box:center+half_box,center-half_box:center+half_box]
				if(np.std(target_filament_lp50 > 0.0001)):
					target_filament_lp50 = (target_filament_lp50 - np.mean(target_filament_lp50)) / np.std(target_filament_lp50) # normalize
				target_lp50[j+1] = target_filament_lp50
			
			
			# Generate noisy image
			target_proj = np.sum(target, axis=0)
			target_proj = (target_proj - np.mean(target_proj)) / np.std(target_proj)
			r7 = local_random_state.uniform(defoc_low, defoc_high) #defocus
			target_eman = EMNumPy.numpy2em(target_proj)

			# Project volume from defined angles and backproject
			r16 = local_random_state.randint(45,66)
			r17= local_random_state.randint(1,5)
			angles = list(range(-1*r16,r16+1,r17))
			projections = []
			for k in range(0,len(angles)):
				t = Transform({'type':'eman','phi': 0, 'alt':angles[k],'az':90})
				temp_box = target_eman.copy()
				projection = temp_box.project('standard', t)
				projection.process_inplace('math.simulatectf',{'ampcont':ampCont,'apix':apix,'bfactor':bfact,'cs':cs,'defocus':r7,'voltage':voltage})
				projection.set_attr('xform.projection', t)
				projections.append(projection)
			
			recon = Reconstructors.get('fourier', {'sym':'c1', 'size':(cropBox*2,cropBox*2,cropBox*2), 'mode':'gauss_2', 'corners':True})
			recon.setup()

			for k in range(0, len(angles)):
				recon.insert_slice(projections[k],projections[k]['xform.projection'], 1.0)

			ret = recon.finish(True)

			target_noise = EMNumPy.em2numpy(ret)

			r14 = local_random_state.normal(0,0.5)
			white_box = np.random.normal(0,0.4,[cropBox*2,cropBox*2,cropBox*2])
			pnk_pw = np.multiply(empirical_pwrSpectrum[r15]+r14*empirical_pwrSpectrum_std[r15], white_box)
			pnk_pw_shift = np.fft.ifftshift(pnk_pw)
			pink_noise_box = np.fft.ifftn(pnk_pw_shift).real
			pink_noise_box = ((pink_noise_box - np.average(pink_noise_box)) / np.std(pink_noise_box))
			target_noise = np.fft.ifftn(np.fft.fftn((np.random.gamma(1.5,0.4)+0.9)*target_noise) + np.fft.fftn(pink_noise_box)).real
			target_noise = target_noise[center-half_box:center+half_box, center-half_box:center+half_box, center-half_box:center+half_box]

			target_noise = (target_noise - np.mean(target_noise)) / np.std(target_noise) # normalize
			with mrcfile.new(noise_outputDir + 'actin_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_noise.astype('float32'))
			
			with mrcfile.new(noNoise_outputDir + 'actin_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_proj.astype('float32'))
			
			
			semMap = np.array(np.zeros((3,cropBox,cropBox,cropBox)))
			semMap[0][target_lp50[1]>0.1] = 1 # actin layer
			semMap[1][target_lp50[2]>0.1] = 1 # fascin layer
			semMap[0][semMap[1]>0.1] = 0 # actin layer
			semMap[2] = -1*(np.maximum(semMap[0], semMap[1]) - 1)
			with mrcfile.new(semMap_outputDir + 'actin_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(semMap[0].astype('float32'))
			
			with mrcfile.new(semMap_outputDir + 'fascin_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(semMap[1].astype('float32'))
			
			with mrcfile.new(semMap_outputDir + 'background_rotated%06d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(semMap[2].astype('float32'))
			

################################################################################
# For Debugging:
#num_per_proc = int(TOTAL_NUM_TO_MAKE / nProcs)
#launch_parallel_process(5)
#sys.exit()

# run in parallel
num_per_proc = int(TOTAL_NUM_TO_MAKE / nProcs)
if __name__ == '__main__':
	p=Pool(nProcs)
	p.map(launch_parallel_process, range(0, nProcs))
	p.close()
	p.join()

print('The program has generated projection images. Compiling metadata...')
################################################################################
# Now all files are written, combine all json files into one master json file
read_files = glob.glob(noise_outputDir+'params_*.json')
output_list = []
for f in read_files:
	for line in open(f, 'r'):
		output_list.append(json.loads(line))

#sort the json dictionaries based on iteration number
output_list = sorted(output_list, key=lambda i: i['iteration'])
for line in output_list:
	with open(noise_outputDir+'master_params.json', 'a') as fp:
		data_to_write = json.dumps(line)
		fp.write(data_to_write + '\n')


print('Finished compiling metadata.')
print('Exiting...')





