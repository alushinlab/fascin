#!/home/alus_soft/matt_EMAN2/bin/python
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
parser.add_argument('--use_empirical_model', type=bool, help='Boolean. Are you using an empirical model? If so, specify the path with that option.')
parser.add_argument('--path_to_empirical_model', type=str, help='String. Relative or absolute path to the empirical model.')
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
	nProcs = args.nProcs

if(args.numProjs % nProcs != 0):
	print('The numProjs that you specified was not a multiple of nProcs.')
	print('Instead of %d 2D projections, this program will generate %d 2D projections'%(args.numProjs, args.numProjs/nProcs*nProcs))

folder = args.input_mrc_dir
noNoise_outputDir = args.output_noiseless_dir
noise_outputDir = args.output_noise_dir
semMap_outputDir = args.output_semMap_dir
TOTAL_NUM_TO_MAKE = args.numProjs
#image sizes
cropBox = args.box_len; half_box = int(cropBox)/2

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
use_empirical_model = args.use_empirical_model
path_to_empirical_model = args.path_to_empirical_model

# bundle parameters
alt_stdev = args.alt_stdev # 7.5
tx_low  = args.tx_low  #-60
tx_high = args.tx_high # 60 
ty_low  = args.ty_low  #-60
ty_high = args.ty_high # 60 
tz_low  = args.tz_low  #-60
tz_high = args.tz_high # 60
max_fil_num = args.max_fil_num # 3
is_bundle = bool(args.is_bundle)     # True
bundle_freq = args.bundle_freq # 0.65
bundle_alt_stdev  = args.bundle_alt_stdev  # 1.5
bundle_type = args.bundle_type # 'Parallel'
bundle_dist = args.bundle_dist # 113.8 A
bundle_dist_stdev = args.bundle_dist_stdev # 15.85 A
bundle_ty_low  = args.bundle_ty_low  # -44
bundle_ty_high = args.bundle_ty_high #  44
if(is_bundle == '0'):
	max_fil_num = 1

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
from skimage.morphology import erosion,dilation; from skimage.morphology import disk
################################################################################
################################################################################
# import data
actin_orig = []
file_names = sorted(os.listdir(folder))
for file_name in file_names:
	if(file_name[-4:] == '.mrc'):
		actin_orig.append(EMData(folder+file_name))

################################################################################
################################################################################
def launch_parallel_process(thread_idx):
	index=num_per_proc*thread_idx
	for i in tqdm(range(0,num_per_proc),file=sys.stdout):
		local_random_state = np.random.RandomState(None)
		# First: randomly pick one of the actin mrc files that were loaded into actin_orig
		r0 = local_random_state.randint(0,len(actin_orig))
		num_filaments = local_random_state.randint(0,max_fil_num+1)
		if(num_filaments == 0):
			target = np.concatenate((np.ones((1,cropBox,cropBox)), np.zeros((2,cropBox,cropBox))), axis=0)
			with mrcfile.new(semMap_outputDir + 'actin_rotated%05d.mrcs'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target.astype('float32'))
			
			r7 = local_random_state.uniform(defoc_low, defoc_high) #defocus default is 1.0, 4.0
			r8 = max(local_random_state.normal(noise_amp, noise_stdev),0) # noise amplitude default is 0.050, 0.010
			target_eman = EMNumPy.numpy2em(np.zeros((cropBox,cropBox)))
			target_eman.process_inplace('math.simulatectf',{'ampcont':ampCont,'apix':apix,'bfactor':bfact,'cs':cs,'defocus':r7,'noiseamp':r8,'purectf':False,'voltage':voltage})
			target_noise = EMNumPy.em2numpy(target_eman)
			with mrcfile.new(noise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_noise.astype('float32'))
			with mrcfile.new(noNoise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target[1].astype('float32'))		

		else:
			target = np.zeros((max_fil_num+2,cropBox,cropBox)); target_lp50 = np.zeros((max_fil_num+2,cropBox,cropBox))
			bundle_idxs = []
			for j in range(1, num_filaments+1):
				r0_name = file_names[r0]
				rotated_actin = actin_orig[r0].copy()
				# handle bundles
				r9 = local_random_state.uniform()
				fil_is_bundle = r9 < bundle_freq
				if(fil_is_bundle): # i.e. if this filament-unit is a bundle
					if(bundle_type == '0'): #0 = Parallel
						r10 = int(local_random_state.normal(loc=0, scale=bundle_alt_stdev)) #
					elif(bundle_type == '1'): # 1 = Antiparallel
						r10 = int(local_random_state.normal(loc=0, scale=bundle_alt_stdev) + 180) #
					elif(bundle_type == '2'): # 2 = Both
						r10 = int(local_random_state.normal(loc=0, scale=bundle_alt_stdev) + int(local_random_state.randint(0,2)*180)) #
					r11 = int(local_random_state.random_sample()*360) #
					r12 = local_random_state.normal(loc=bundle_dist, scale=bundle_dist_stdev) # bundle distance, measured by Alfred
					r13 = local_random_state.uniform(bundle_ty_low,bundle_ty_high) # should cover full crossover
					rotated_actin2 = actin_orig[r0].copy()
					t2 = Transform()
					t2.set_params({'type':'eman', 'az':0, 'alt':r10, 'phi':r11, 'tx':0, 'ty':r12, 'tz':r13}) #tz is shift up and down filament axis # y and x control distance, alt is (anti-)parallel
					rotated_actin2.transform(t2)
					rotated_actin.add(rotated_actin2)
					bundle_idxs.append(j)
				
				# Rotation angles: azimuth, alt, phi, then Translations: tx, ty,tz
				r1, r2, r3 = int(local_random_state.random_sample()*360),int(local_random_state.normal(loc=90, scale=alt_stdev)),int(local_random_state.random_sample()*360)
				#r4, r5, r6 = local_random_state.normal(0, 25), local_random_state.normal(0, 25), local_random_state.normal(0, 25)
				r4, r5, r6 = local_random_state.uniform(tx_low, tx_high), local_random_state.uniform(ty_low, ty_high), local_random_state.uniform(tz_low, tz_high)
				t = Transform()
				t.set_params({'type':'eman','az':r1, 'alt':r2, 'phi':r3, 'tx':r4, 'ty':r5, 'tz':r6})
				rotated_actin.transform(t) # apply rotation and translation
				proj_eman = rotated_actin.project('standard',Transform()) # project
				proj_eman_lp50 = proj_eman.process('filter.lowpass.gauss', {'apix':apix, 'cutoff_freq':1.0/lowpass_res})
				proj_np_lp50 = EMNumPy.em2numpy(proj_eman_lp50)
				proj_np_lp50 = dilation(proj_np_lp50, selem=disk(dilation_radius))
				proj_np_lp50 = erosion(proj_np_lp50, selem=disk(erosion_radius))
				proj_np = EMNumPy.em2numpy(proj_eman)
				center = cropBox
				# Save the target image
				target_filament = proj_np[center-half_box:center+half_box, center-half_box:center+half_box]
				#target_filament = (target_filament - np.mean(target_filament)) / np.std(target_filament) # normalize
				target[j] = target_filament
				
				target_filament_lp50 = proj_np_lp50[center-half_box:center+half_box, center-half_box:center+half_box]
				target_filament_lp50 = (target_filament_lp50 - np.mean(target_filament_lp50)) / np.std(target_filament_lp50) # normalize
				target_lp50[j] = target_filament_lp50
			
			
			# Generate noisy image
			target_proj = np.sum(target, axis=0)
			target_proj = (target_proj - np.mean(target_proj)) / np.std(target_proj)
			r7 = local_random_state.uniform(defoc_low, defoc_high) #defocus
			r8 = max(local_random_state.normal(noise_amp, noise_stdev),0) # noise amplitude 0.050, 0.010 is default perhaps also try 0.027 0.005
			target_eman = EMNumPy.numpy2em(target_proj)
			target_eman.process_inplace('math.simulatectf',{'ampcont':ampCont,'apix':apix,'bfactor':bfact,'cs':cs,'defocus':r7,'noiseamp':r8,'purectf':False,'voltage':voltage})
			target_noise = EMNumPy.em2numpy(target_eman)
			target_noise = (target_noise - np.mean(target_noise)) / np.std(target_noise) # normalize
			with mrcfile.new(noise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_noise.astype('float32'))
			with mrcfile.new(noNoise_outputDir + 'actin_rotated%05d.mrc'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(target_proj.astype('float32'))
			
			# make masks from each proj
			for j in range(1, num_filaments+1):
				target_lp50[j] = (target_lp50[j]>0)
			temp = np.sum(target_lp50, axis=0)
			
			semMap = np.array(np.zeros((3,cropBox,cropBox)))
			semMap[0] = (temp==0)>0
			#semMap[1] = -1*semMap[0] + 1
			semMap[2] = np.sum(target_lp50[bundle_idxs], axis=0) > 0
			semMap[1] = -1*((semMap[0] + semMap[2]) - 1)
			with mrcfile.new(semMap_outputDir + 'actin_rotated%05d.mrcs'%(i+num_per_proc*thread_idx), overwrite=True) as mrc:
				mrc.set_data(semMap.astype('float32'))


################################################################################
# run in parallel
num_per_proc = TOTAL_NUM_TO_MAKE / nProcs
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





