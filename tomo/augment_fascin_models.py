#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import string
import glob
from tqdm import tqdm
import mrcfile
print('Imports finished. Beginning script...')
################################################################################
TOTAL_SAMPLES = 200
VOXEL_SIZE = 7.8
BOX_SIZE = 128
################################################################################
def last_uppercase_letter(s):
    for i in range(0, len(s)):
        if s[i].isupper():
            return s[i]
    return None

# load volumes
filament_names = sorted(glob.glob('./mrcs/*_long.mrc'))
filament_holder = {}
for i in range(0, len(filament_names)):
    with mrcfile.open(filament_names[i], 'r') as mrc:
        mrc = mrc.data
    filament_holder[last_uppercase_letter(filament_names[i])] = mrc

fascin_names = sorted(glob.glob('./mrcs/fascin*.mrc'))
fascin_holder = {}
for i in range(0, len(fascin_names)):
    with mrcfile.open(fascin_names[i], 'r') as mrc:
        mrc = mrc.data
    fascin_holder[fascin_names[i]] = mrc



################################################################################
actin_keys = sorted(set(string.ascii_uppercase[:19]))
fascin_keys = sorted(set(['AB','AD','AE','BC','BE','BF','CF','CF','CG','DA','DE','DH','DI','EF','EI',
               'EJ','FG','FJ','FK','GK','GL','HI','HM','IJ','IM','IN','JK','JN','JO','KL',
               'KP','LP','MN','MQ','NO','NQ','NR','OP','OR','OS','PS','QR','RS']))



# randomly sample subset of actin keys, with weights
actin_prob_weights = np.arange(0,len(actin_keys))
actin_prob_weights = actin_prob_weights / (actin_prob_weights.sum()*1.0)

num_filaments = []
for i in range(0, TOTAL_SAMPLES):
    num_filaments.append(np.random.choice(len(actin_keys), p=actin_prob_weights))

which_filaments = []
for i in range(0, TOTAL_SAMPLES):
    which_filaments.append(sorted(np.random.choice(actin_keys, num_filaments[i],replace=False)))

kept_matching_fascin_keys = []
for i in range(0,TOTAL_SAMPLES):
    matching_fascin_keys = []
    for j in range(0, len(which_filaments[i])):
        for k in range(0, len(fascin_keys)):
            if(which_filaments[i][j] in fascin_keys[k]):
                matching_fascin_keys.append(fascin_keys[k])
    
    frac_to_keep = np.random.random()
    np.random.shuffle(matching_fascin_keys)
    matching_fascin_keys = matching_fascin_keys[:int(len(matching_fascin_keys)*frac_to_keep)]
    kept_matching_fascin_keys.append(matching_fascin_keys)

print('Saving volumes...')
for i in tqdm(range(0, TOTAL_SAMPLES)):
    actin_filaments = np.zeros((BOX_SIZE,BOX_SIZE,BOX_SIZE))
    for j in range(0, len(which_filaments[i])):
        actin_filaments = np.maximum(actin_filaments,filament_holder[which_filaments[i][j]])
    
    with mrcfile.new('./mrc_composites/filaments_only_'+str(i).zfill(4) + '.mrc', overwrite=True) as mrc:
        mrc.set_data(actin_filaments.astype('float32'))
        mrc.voxel_size = (VOXEL_SIZE,VOXEL_SIZE,VOXEL_SIZE)
        mrc.flush()
    
    fascin_volume = np.zeros((BOX_SIZE,BOX_SIZE,BOX_SIZE))
    for j in range(0, len(kept_matching_fascin_keys[i])):
        file_name = [key for key in fascin_holder.keys() if kept_matching_fascin_keys[i][j] in key]
        for k in range(0, len(file_name)):
            if(np.random.random() <= 0.8):
                fascin_volume = np.maximum(fascin_volume,fascin_holder[file_name[k]])
    
    with mrcfile.new('./mrc_composites/fascin_only_'+str(i).zfill(4) + '.mrc', overwrite=True) as mrc:
        mrc.set_data(fascin_volume.astype('float32'))
        mrc.voxel_size = (VOXEL_SIZE,VOXEL_SIZE,VOXEL_SIZE)
        mrc.flush()

print('Finished saving volumes. Exiting...')
