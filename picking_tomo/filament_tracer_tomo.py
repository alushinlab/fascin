#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
import argparse; import sys
parser = argparse.ArgumentParser('Trace filaments through a tomogram')
# IO parameters
parser.add_argument('--input_mrc', type=str, help='Input file containing the tomogram in an MRC format')
parser.add_argument('--output_dir', type=str, help='Directory to store filament traces')
# Other parameters
parser.add_argument('--cyl_diam', type=float, help='diameter of cylinder used for computing cross-correlation')
parser.add_argument('--box_size', type=int, help='size of box for calculating cross-correlation')
parser.add_argument('--angpix', type=float, help='Angstroms per pixel in input tomogram')
parser.add_argument('--cone_opening', type=float, help='Bfactor')
parser.add_argument('--cone_height', type=float, help='Spherical Aberration')
parser.add_argument('--c_param', type=float, help='co-circularity parameter')
parser.add_argument('--l_param', type=float, help='linearity parameter)')
parser.add_argument('--d_param', type=float, help='distance parameter, in angstroms)')
parser.add_argument('--t1_threshold', type=float, help='t1 influences the length of filaments traced')
parser.add_argument('--t2_threshold', type=float, help='t2 influences the number of filaments traced (min. correlation)')
args = parser.parse_args()
print('')
if(args.input_mrc == None or args.output_dir == None):
	print('Please enter an input_mrc, AND an output_dir')
	sys.exit('The preferred input style may be found with ./filament_tracer_tomo.py -h')

if(args.cyl_diam == None or args.box_size == None or args.angpix == None or args.cone_opening == None or args.cone_height == None 
	or args.c_param == None or args.l_param == None or args.d_param == None or args.t1_threshold == None or args.t2_threshold == None):
		sys.exit('Please enter all input parameters.')

input_mrc_name = args.input_mrc
outputDir = args.output_dir
cyl_diam = args.cyl_diam
box_size = args.box_size
angpix = args.angpix
cone_opening = args.cone_opening
cone_height = args.cone_height
c_param = args.c_param
l_param = args.l_param
d_param = args.d_param
t1_threshold = args.t1_threshold
t2_threshold = args.t2_threshold


if(outputDir[-1] != '/'): outputDir = outputDir + '/'
print('Inputs accepted.')
print('Importing packages...')
################################################################################
# import of python packages
import numpy as np
import healpy as hp
import warnings
from EMAN2 import *; from sparx import *; import mrcfile
import json; import glob
from multiprocessing import Pool
import os; from tqdm import tqdm
from skimage.morphology import erosion,dilation; from skimage.morphology import disk, ball
################################################################################
# Generate cylinder volume
# adapted from: https://github.com/cryoem/eman2/blob/master/examples/e2cylinder.py
def cylinder(box, radius, height, apix):
	mask = EMData(box, box, box)
	mask.to_one()
	maskout = mask.process("testimage.cylinder",{'height':height,'radius':radius})
	#maskout_lp = maskout.process('filter.lowpass.gauss', {'apix':apix, 'cutoff_freq':1.0/40})
	#finalmask = maskout_lp.process('mask.soft', {'value':0.9, 'width':0.1})
	#finalmask = maskout_lp.process('threshold.binary', {'value':0.8})
	finalmask = maskout
	finalmask['apix_x']=apix
	finalmask['apix_y']=apix
	finalmask['apix_z']=apix
	return finalmask

def healpix_sampling(nside):
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # Convert to quaternion
    stheta = np.sin(theta/2)
    x = stheta * np.cos(phi)
    y = stheta * np.sin(phi)
    z = np.cos(theta/2)
    # quaternions are usually represented as w + xi + yj + zk, so we'll prepend a zero column for w
    quaternions = np.column_stack((np.zeros(npix), x, y, z))
    v = np.array([1,0,0])
    rotated_vectors = np.array([quaternion_rotate(v, q) for q in quaternions])

    return quaternions, rotated_vectors

def quaternion_rotate(v, q):
    """
    Rotate 3D-vector v using quaternion q.
    """
    # quaternion multiplication: q*v*q'
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])  # conjugate of q
    w = np.array([0, v[0], v[1], v[2]])  # convert v into a quaternion with real part 0
    t = quaternion_mult(q, quaternion_mult(w, q_conj))
    return np.array([t[1], t[2], t[3]])

def quaternion_mult(q1, q2):
    """
    Multiply two quaternions.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

from scipy import signal, fftpack
def compute_cross_correlation(tomo, template):
    # Make sure that tomo is larger than template
    assert np.all(np.array(tomo.shape) >= np.array(template.shape))
    # Pad template to match the shape of tomo
    pad_shape = np.array(tomo.shape) - np.array(template.shape)
    template_padded = np.pad(template, [(0, pad_shape[dim]) for dim in range(template.ndim)])
    # Perform FFT on both tomo and template
    tomo_fft = np.fft.fftn(tomo)
    template_fft = np.fft.fftn(template_padded)
    # Compute the cross-correlation through the inverse FFT of the product of FFTs
    # Note that we are using FFT shift to ensure the zero frequency component is centered
    cross_corr = np.abs(np.fft.ifftn(tomo_fft * template_fft.conj()))
    return cross_corr

################################################################################
# import data
print('All packages imported.')
print('The program will now trace filaments through the tomogram...')

print('Loading tomo into RAM...')
tomo = EMData(input_mrc_name)
print('Tomo loaded successfully.')
print('Generating cylinder templates...')
probe_cyl = cylinder(box_size, cyl_diam/angpix, box_size, angpix)
probe_cyl_np = EMNumPy.em2numpy(probe_cyl)
with mrcfile.new(outputDir + 'probe_cylinder.mrc', overwrite=True) as mrc:
	mrc.set_data(probe_cyl_np.astype('float32'))
	mrc.voxel_size = angpix

template_sampling_resolution = 3
euler_angles, rotated_vectors = healpix_sampling(template_sampling_resolution)
euler_angles = euler_angles[:int(len(euler_angles)/2)]
rotated_vectors = rotated_vectors[:int(len(euler_angles)/2)]

template_holder = []
template_holder_np = []
for i in range(0, len(euler_angles)):
	template = probe_cyl.copy()
	t = Transform()
	t.set_params({'type':'quaternion','e0':euler_angles[i][0], 'e1':euler_angles[i][1], 'e2':euler_angles[i][2],'e3':euler_angles[i][3]})
	template.transform(t) # apply rotation and translation
	template_holder.append(template)
	template_holder_np.append(EMNumPy.em2numpy(template))
	#template_np = EMNumPy.em2numpy(template)
	#with mrcfile.new(outputDir + 'template%s.mrc'%(str(i).zfill(3)), overwrite=True) as mrc:
	#	mrc.set_data(template_np.astype('float32'))
	#	mrc.voxel_size = angpix

print('HEALPix nside sampling: ' +str(template_sampling_resolution))
print('Total number of cylindrical templates generated: ' + str(euler_angles.shape[0]))
print('Computing CCF between tomogram and templates...')
tomo_np = EMNumPy.em2numpy(tomo)
tomo_np = tomo_np[:,300:800,300:800]
with mrcfile.new(outputDir + 'croppedInput.mrc', overwrite=True) as mrc:
	mrc.set_data(tomo_np.astype('float32'))
	mrc.voxel_size = angpix
template_holder_np = np.asarray(template_holder_np)
CCF_holder = np.zeros((len(template_holder_np),tomo_np.shape[0], tomo_np.shape[1], tomo_np.shape[2]))

from multiprocessing import Pool
def worker(args):
    i, template = args
    c = compute_cross_correlation(tomo_np, template)
    return i, c

warnings.filterwarnings("ignore", module="json")
if __name__ == "__main__":
    N = 6  # replace with your desired number of cores
    n_templates = len(template_holder_np)
    CCF_holder = np.zeros((n_templates, tomo_np.shape[0], tomo_np.shape[1], tomo_np.shape[2]))
    with Pool(N) as pool:
        with tqdm(total=n_templates) as pbar:
            for i, result in pool.imap_unordered(worker, enumerate(template_holder_np)):
                CCF_holder[i] = result
                pbar.update()

#for i in tqdm(range(0, len(template_holder_np))):
#	c = compute_cross_correlation(tomo_np, template_holder_np[i])
#	CCF_holder[i] = c

print('Finished computing CCF between tomogram and all templates.')
print('Combining CCF maps into maximum value...')
print(CCF_holder.shape)
temp = np.min(CCF_holder, axis=0)
print(temp.shape)
with mrcfile.new(outputDir + 'CCC.mrc', overwrite=True) as mrc:
	mrc.set_data(temp.astype('float32'))
	mrc.voxel_size = angpix


sys.exit()




CCF_holder = []
for i in tqdm(range(0, 3)):#len(template_holder))):
	c = tomo.calc_ccf(template_holder[i])
	c_np = EMNumPy.em2numpy(c)
	c_np = np.fft.fftshift(c_np).real
	CCF_holder.append(c_np)

CCF_holder = np.asarray(CCF_holder)
print(CCF_holder.shape)
temp = np.max(CCF_holder, axis=0)
print(temp.shape)
with mrcfile.new(outputDir + 'CCC.mrc', overwrite=True) as mrc:
	mrc.set_data(temp.astype('float32'))
	mrc.voxel_size = angpix




sys.exit()
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





