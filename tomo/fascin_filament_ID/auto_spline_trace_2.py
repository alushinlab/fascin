#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import mrcfile
import glob
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, ball
from skimage.morphology import skeletonize_3d, binary_dilation
from scipy.ndimage import convolve
from scipy.spatial import distance_matrix
import os
print('Imports finished. Beginning script...')
################################################################################
def mask_z_cylinder(shape, center, radius, the_map):
    z,y,x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    masked_result = np.multiply(mask, the_map)
    mask =  np.multiply(mask, np.ones_like(the_map))
    return masked_result, mask

def find_endPoints(skeleton):
    # Define a kernel
    s = np.ones((3, 3, 3))
    s[1,1,1] = 30
    neighbors_count = convolve(skeleton.astype(int), s)
    # Find voxels that are on and have more than 2 on-neighbors
    end_points = (skeleton == 1) & (neighbors_count == 31)
    return end_points

def prune_nubs(skel, threshold_length=10):
    endpoints = find_endPoints(skel)
    print('Endpoints identified: ' + str(np.sum(endpoints)))
    visited = np.zeros_like(skel, dtype=bool)
    pruned_skel = np.copy(skel)        
    for endpoint in np.argwhere(endpoints):
        current = tuple(endpoint)
        branch_length = 0
        path = [current]
        while True:
            visited[current] = True
            neighbors = find_neighbors(current, skel)
            neighbors = [n for n in neighbors if not visited[n]]
            # If this is a branch point or another endpoint
            if len(neighbors) != 1:
                break
            current = neighbors[0]
            path.append(current)
            branch_length += 1
        # If the branch length is less than the threshold, prune the branch
        if branch_length < threshold_length:
            for point in path:
                pruned_skel[point] = False

    print('Endpoints after pruning: ' + str(np.sum(find_endPoints(pruned_skel))))
    return pruned_skel

def find_neighbors(point, skel):
    offsets = np.array([ 
            [i, j, k] 
            for i in [-1, 0, 1] 
            for j in [-1, 0, 1] 
            for k in [-1, 0, 1]
            if (i, j, k) != (0, 0, 0)
        ])
        
    neighbors = []
    for offset in offsets:
        neighbor = tuple(point + offset)
        if skel[neighbor]:
            neighbors.append(neighbor)
    return neighbors

def find_triplePoints(skeleton):
    # Define a kernel
    s = np.ones((3, 3, 3))
    s[1,1,1] = 30
    neighbors_count = convolve(skeleton.astype(int), s)
    # Find voxels that are on and have more than 2 on-neighbors
    branch_points = (skeleton == 1) & (neighbors_count == 33)
    return branch_points

def trace_branches_from_triple(skel, start_point):
    branches, branches_lengths = [], []
    visited = np.zeros_like(skel, dtype=bool)
    initial_neighbors = find_neighbors(start_point, skel)
    visited[start_point[0], start_point[1], start_point[2]] = True
    for neighbor in initial_neighbors:
        if visited[neighbor]:
            continue
        branch = [start_point]
        current = neighbor
        while True:
            branch.append(current)
            visited[current] = True
            neighbors = find_neighbors(current, skel)
            neighbors = [n for n in neighbors if not visited[n]]
            # If endpoint or another triple point is found, break
            if len(neighbors) != 1:
                break
            current = neighbors[0]
        branches.append(branch)
        branches_lengths = [len(branch) for branch in branches]
    return branches, branches_lengths

def remove_shortest_branch(skel, branches, branches_lengths):
    if not branches_lengths:
        return skel
    # Find the index of the shortest branch
    shortest_branch_idx = branches_lengths.index(min(branches_lengths))
    # Remove the shortest branch from the skeleton
    for point in branches[shortest_branch_idx][1:]:
        coord_z, coord_y, coord_x = point
        skel[coord_z, coord_y, coord_x] = 0
    return skel

def apply_skel(closed_img, prune_length):
    skel = skeletonize_3d(closed_img)>0.95
    skel = prune_nubs(skel, prune_length)
    branch_points = find_triplePoints(skel)
    branch_points_xyz = np.argwhere(branch_points)
    for i in range(0, len(branch_points_xyz)):
        branches, branch_lengths = trace_branches_from_triple(skel, branch_points_xyz[i])
        if(branch_lengths != 1):
            skel = remove_shortest_branch(skel, branches, branch_lengths)
        else:
            if(branch_lengths[0] > 100):
                continue
            else:
                skel = remove_shortest_branch(skel, branches, branch_lengths)
    return skel

def trace_branch_from_endpoint(skel, start_point):
    branch = [start_point]
    visited = np.zeros_like(skel, dtype=bool)
    current = start_point
    visited[current[0],current[1],current[2]]  = True
    while True:
        neighbors = find_neighbors(current, skel)
        neighbors = [n for n in neighbors if not visited[n]]
        # If an endpoint or a branching point is found, break
        if len(neighbors) != 1:
            break
        current = neighbors[0]
        branch.append(current)
        visited[current] = True
    return branch

def branches_are_same(branch1, branch2):
    return np.array_equal(branch1, branch2) or np.array_equal(branch1, branch2[::-1])

def remove_duplicate_branches(branches_list):
    unique_branches = []
    for i in range(len(branches_list)):
        branch = branches_list[i]
        is_duplicate = False
        for j in range(i + 1, len(branches_list)):
            if branches_are_same(branch, branches_list[j]):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_branches.append(branch)
    return unique_branches

def remove_connectors(ordered_points, threshold_distance=15):  # adjust the threshold if needed 12 for this and 15 for range works
    distances = distance_matrix(ordered_points, ordered_points)
    np.fill_diagonal(distances, np.inf)  # ignore self-distances
    mid_idx = len(ordered_points) // 2
    adj_range = set(range(-22, 23))  # Using a set for O(1) membership checks
    for offset in range(1, mid_idx + 1):
        for i in [mid_idx + offset, mid_idx - offset]:  # Check both sides from the midpoint
            if i >= len(ordered_points):
                continue
            close_points = np.where(distances[i] < threshold_distance)[0]  # indices of close points
            for j in close_points:
                # Check for non-adjacency
                if j - i not in adj_range:
                    start, end = sorted([i, j])
                    midpoint = (start + end) // 2
                    # Splitting into two filaments
                    filament_1 = ordered_points[:midpoint]
                    filament_2 = ordered_points[midpoint:]
                    return [filament_1, filament_2]
    return [ordered_points]

def generate_distinct_colors(n):
    """Generate a list of distinct RGB colors."""
    np.random.seed(42)  # Setting seed for reproducibility
    colors = np.random.rand(n, 3)
    return colors

def save_to_cmm(file_name, points, color):
    """Save points to a .cmm file with the specified color."""
    with open(file_name, 'w') as file:
        file.write('<marker_set name="{}">\n'.format(os.path.basename(file_name).split('.')[0]))
        for idx, (z, y, x) in enumerate(points):
            file.write('<marker id="{}" x="{}" y="{}" z="{}" r="{}" g="{}" b="{}" radius="3"/>\n'.format(
                idx + 1, x, y, z, color[0], color[1], color[2]))
        file.write('</marker_set>\n')

################################################################################
################################################################################
# Center of carbon holes, manually measured, X,Y
cylinders = [[910,840],[500,910],[740,980],[-10,660],[580,1170],[570,1040],[555,765],
	     	 [610,980],[570,980],[550,920],[570,1000],[490,990],[530,1029],[590,1010],
			 [570,1030],[540,1000],[590,1080],[540,920],[550,980],[630,1020],[540,1000],
			 [610,1000],[560,1020],[920,300],[-200,790],[660,1040]]
RADIUS_PX = 740
################################################################################
#base_file_names = ['/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/semSeg_actin/ts059.st_rec_channel0.mrc']
base_file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/semSeg_actin/*.mrc'))
masked_file_name_location = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/semSeg_actin_masked/'
for i in range(0, len(base_file_names)):
    file_name = base_file_names[i].split('/')[-1]
    # Load in image data; denoised tomogram
    print('Loading in denoised map ' +file_name+'...')
    denoised_file_name =  base_file_names[i]
    with mrcfile.open(denoised_file_name) as mrc:
        denoised_map = mrc.data.astype('float32')
    print('Denoised map loaded.')
    print('Masking map...')
    masked_semSeg, z_mask = mask_z_cylinder(denoised_map.shape, cylinders[i], RADIUS_PX, denoised_map)
    print('Lowpass filtering and removing small blobs...')
    masked_semSeg_lp = gaussian_filter(masked_semSeg, sigma=2)
    masked_semSeg_cleaned = remove_small_objects(masked_semSeg_lp>0.7, min_size=500)
    with mrcfile.new(masked_file_name_location + file_name, overwrite=True) as mrc:
        mrc.set_data(masked_semSeg_cleaned.astype('float32'))
        mrc.voxel_size = (3.0,3.0,3.0)
    
    print('Map masked, blobs removed, and saved.')
    print('Skeletonizing, pruning nubs, and removing short branches...')
    cleaned_up_skeleton = apply_skel(masked_semSeg_cleaned, 30)
    dilated_img = binary_dilation(cleaned_up_skeleton, ball(3))
    print('Image processed. Saving...')
    with mrcfile.new(masked_file_name_location + file_name[:-4]+'_dilated.mrc', overwrite=True) as mrc:
        mrc.set_data(dilated_img.astype('float32'))
        mrc.voxel_size = (3.0,3.0,3.0)
    
    print('Tracing filaments in skeleton...')
    endpoints = np.argwhere(find_endPoints(cleaned_up_skeleton))
    all_ordered_points = []
    for j in tqdm(range(0, len(endpoints))):
        branch = trace_branch_from_endpoint(cleaned_up_skeleton, endpoints[j])
        if(len(branch) > 50):
            all_ordered_points.append(branch)

    print('Removing duplicate branches...')
    all_ordered_points = remove_duplicate_branches(all_ordered_points)
    print('Breaking any looped filament pairs...')
    all_cleaned_points = []
    for j in tqdm(range(0, len(all_ordered_points))):
        cleaned_points = remove_connectors(all_ordered_points[j])
        for k in range(0, len(cleaned_points)):
            if(len(cleaned_points[k]) > 50):
                all_cleaned_points.append(cleaned_points[k])

    print('Finished processing. Saving cmm files...')
    # Save cmm file
    output_dir = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/automated_fil_identification/'
    output_dir = output_dir + file_name.split('.')[0] + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    colors = generate_distinct_colors(len(all_cleaned_points))
    for j in tqdm(range(0, len(all_cleaned_points))):
        cmm_file_name = output_dir+ 'set_%s.cmm'%str(j).zfill(3)
        save_to_cmm(cmm_file_name, (3.0*np.asarray(all_cleaned_points[j]))[::5], colors[j])


# TODO: save all_cleaned_points as cmm file; also make a subsetted one to every 10 points or so for the spline fitting
print('Hi')




