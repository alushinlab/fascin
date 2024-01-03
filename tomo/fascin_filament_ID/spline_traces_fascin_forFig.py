#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import mrcfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter
import sys
from scipy.spatial.distance import cdist
import networkx as nx
from networkx.algorithms.community import girvan_newman
import pickle
#import community
print('Imports finished. Beginning script...')
################################################################################
def load_cmm_data(file_name):
    text_holder = np.genfromtxt(file_name, delimiter='\"', dtype=str, skip_header=2, skip_footer=1)
    cmm_data = text_holder[:,[3,5,7]]
    return cmm_data

def load_bild_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()[2:]  # skip the first line

    # Extract every other line after the first
    needed_lines = lines[::2]

    # Split the line and extract the desired columns
    bild_data = [line.split()[1:4] for line in needed_lines]
    return np.asarray(bild_data).astype(float)

def create_spline(initial_points, eval_points, s, k=2):
    # Generate a range of x values corresponding to the number of points
    x_values = np.arange(len(initial_points))

    # Create a univariate spline for each dimension (x, y, z)
    splines = []
    for dim in range(3):  # x, y, z
        y_values = initial_points[:, dim]
        spline = UnivariateSpline(x_values, y_values, s=s, k=k)
        splines.append(spline)

    # Compute the arc length of the spline
    arc_lengths = [0]
    for i in range(1, len(x_values)):
        delta_s = np.sqrt(sum((splines[dim](x_values[i]) - splines[dim](x_values[i-1]))**2 for dim in range(3)))
        arc_lengths.append(arc_lengths[-1] + delta_s)

    total_arc_length = arc_lengths[-1]
    target_arc_lengths = np.linspace(0, total_arc_length, eval_points)

    # Now find the x values that correspond to these target arc lengths
    x_eval = np.interp(target_arc_lengths, arc_lengths, x_values)
    # Evaluate the splines at the computed x values
    curve_points = np.vstack([spline(x_eval) for spline in splines]).T
    return curve_points, total_arc_length

def bin_volume_avg(volume, bin_factor):
    # Determine the shape to which the volume can be resized (divisible by the bin_factor)
    new_shape = (volume.shape[0] // bin_factor * bin_factor, 
                 volume.shape[1] // bin_factor * bin_factor, 
                 volume.shape[2] // bin_factor * bin_factor)

    # Slice the volume to this new shape
    sliced_volume = volume[:new_shape[0], :new_shape[1], :new_shape[2]]

    # Define the shape needed for binning
    shape = (sliced_volume.shape[0]//bin_factor, bin_factor,
             sliced_volume.shape[1]//bin_factor, bin_factor,
             sliced_volume.shape[2]//bin_factor, bin_factor)
    
    return sliced_volume.reshape(shape).mean(-1).mean(1).mean(1)

def create_volume_from_spline(curve_points, radius, map_shape):
    volume = np.zeros(map_shape)
    for point in curve_points:
        # Create a mask for points within the sphere/cylinder
        z, y, x = np.ogrid[-radius: radius+1, -radius: radius+1, -radius: radius+1]
        mask = x**2 + y**2 + z**2 <= radius**2

        # Calculate bounds for placing mask onto the volume
        z_start, z_end = int(max(0, point[2] - radius)), int(min(map_shape[0], point[2] + radius + 1))
        y_start, y_end = int(max(0, point[1] - radius)), int(min(map_shape[1], point[1] + radius + 1))
        x_start, x_end = int(max(0, point[0] - radius)), int(min(map_shape[2], point[0] + radius + 1))

        # Calculate bounds for the mask
        mz_start = int(radius - (point[2] - z_start))
        mz_end = int(mz_start + (z_end - z_start))

        my_start = int(radius - (point[1] - y_start))
        my_end = int(my_start + (y_end - y_start))

        mx_start = int(radius - (point[0] - x_start))
        mx_end = int(mx_start + (x_end - x_start))

        # Clip mask and volume slices to ensure they have the same shape
        mask_region = mask[mz_start:mz_end, my_start:my_end, mx_start:mx_end]
        volume_region = volume[z_start:z_end, y_start:y_end, x_start:x_end]

        min_depth = min(mask_region.shape[0], volume_region.shape[0])
        min_height = min(mask_region.shape[1], volume_region.shape[1])
        min_width = min(mask_region.shape[2], volume_region.shape[2])

        # Assign values using the clipped mask and volume regions
        volume_region[:min_depth, :min_height, :min_width][mask_region[:min_depth, :min_height, :min_width]] = 1

    volume = gaussian_filter(volume, sigma=1)  # Adjust sigma as needed
    return volume

def compute_spline_and_volume(data_points, denoised_map, file_name):
	map_shape = denoised_map.shape
	print('Creating uniformly sampled spline...')
	_, arc_length = create_spline(data_points, len(data_points), DEFAULT_S, 1)
	print(arc_length)
	NUM_POINTS = int(arc_length / 7.6)
	print('Using ' + str(NUM_POINTS) + ' points for the calculation.')
	curve_points, _ = create_spline(data_points, NUM_POINTS*9, DEFAULT_S, 1)
	volume_optimized = create_volume_from_spline(curve_points, 2, map_shape)
	BIN_FACTOR=1
	#volume_optimized = bin_volume_avg(volume_optimized, BIN_FACTOR)
	with mrcfile.new(file_name, overwrite=True) as mrc:
		mrc.set_data(volume_optimized.astype('float32'))
		mrc.voxel_size = (3.0*BIN_FACTOR,3.0*BIN_FACTOR,3.0*BIN_FACTOR)
    
	return volume_optimized, curve_points

def compute_spline_only(data_points, OVER_SAMPLING):
	_, arc_length = create_spline(data_points, len(data_points), DEFAULT_S, 1)
	NUM_POINTS = int(arc_length / 7.6)
	curve_points, _ = create_spline(data_points, NUM_POINTS*OVER_SAMPLING, DEFAULT_S, 1)
	return curve_points

def find_two_closest_splines(point, curve_points_holder, cutoff_distance):
    distances = [np.min(cdist(point.reshape(1, -1), spline)) for spline in curve_points_holder]
    two_closest = sorted(range(len(distances)), key=lambda k: distances[k])[:2]
    if distances[two_closest[0]] > cutoff_distance or distances[two_closest[1]] > cutoff_distance:
        return None
    return tuple(sorted(two_closest))

def group_points_by_splines(points, curve_points_holder, cutoff_distance):
    # A dictionary where keys are tuple of two closest spline indices and values are lists of points
    groups = {}
    rejected = []
    for point in tqdm(points):
        closest_splines = find_two_closest_splines(point, curve_points_holder, cutoff_distance)
        if closest_splines is None:
            rejected.append(point)
            continue
        splines = frozenset(closest_splines)
        splines = tuple(splines)
        if splines not in groups:
            groups[splines] = []
        groups[splines].append(point)
    return groups, rejected

def order_points_along_path(points, spline1, spline2):
    # Identify shorter and longer spline
    if len(spline1) <= len(spline2):
        shorter_spline, longer_spline = spline1, spline2
    else:
        shorter_spline, longer_spline = spline2, spline1
    # Compute distances between each point and the spline
    distances = cdist(points, longer_spline)
    closest_spline_indices = np.argmin(distances, axis=1)
    # Sort the points based on their closest point indices on the spline
    ordered_indices = np.argsort(closest_spline_indices)
    ordered_points = np.array([points[i] for i in ordered_indices])
    return ordered_points

def generate_bild(o, data, color, avg_energy):
    avg_color = np.mean(color, axis=0)#np.asarray(color)#np.mean(color, axis=0)
    out=open(o, 'w')
    out.write('.comment ' + f"{avg_energy:.1f} \n")
    out.write('.comment ' + f"{(avg_color*255.0)[0]:.1f}" + ' ' +f"{(avg_color*255.0)[1]:.1f}" + ' ' + f"{(avg_color*255.0)[2]:.1f} \n")
    out.write('.transparency 0.0\n') # 1.0 is fully transparent, 0.0 is opaque
    for j in range(0, len(data)):
        #write out marker entries for each residue pair
        #out.write('.color %.5f %.5f %.5f\n'%(avg_color[0], avg_color[1], avg_color[2])) # color all same color
        out.write('.color %.5f %.5f %.5f\n'%(color[j][0], color[j][1], color[j][2])) # color individually
        out.write(".sphere %.5f %.5f %.5f %.5f \n"%(data[j][0], data[j][1], data[j][2], 6))
    #write final line of xml file, is constant	
    out.close()

def generate_OBB(p1, p2, width, height):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    # Find the line direction
    direction = p2 - p1
    direction /= np.linalg.norm(direction)
    # Find two orthogonal directions
    orthogonal_1 = np.cross(direction, [1, 0, 0])
    if np.linalg.norm(orthogonal_1) < 1e-5:
        orthogonal_1 = np.cross(direction, [0, 1, 0])
    orthogonal_1 /= np.linalg.norm(orthogonal_1)
    orthogonal_2 = np.cross(direction, orthogonal_1)
    orthogonal_2 /= np.linalg.norm(orthogonal_2)
    # Scale orthogonal directions by half-width and half-height
    orthogonal_1 *= width / 2
    orthogonal_2 *= height / 2
    # Find the 8 corners of the OBB
    corners = [
        p1 - orthogonal_1 - orthogonal_2,
        p1 - orthogonal_1 + orthogonal_2,
        p1 + orthogonal_1 - orthogonal_2,
        p1 + orthogonal_1 + orthogonal_2,
        p2 - orthogonal_1 - orthogonal_2,
        p2 - orthogonal_1 + orthogonal_2,
        p2 + orthogonal_1 - orthogonal_2,
        p2 + orthogonal_1 + orthogonal_2
    ]
    return corners

def is_inside_OBB(point, obb_corners):
    # Define the axes based on OBB corners
    axes = [
        obb_corners[1] - obb_corners[0],
        obb_corners[2] - obb_corners[0],
        obb_corners[4] - obb_corners[0]
    ]

    V = point - obb_corners[0]

    for axis in axes:
        projection_length = np.dot(V, axis) / np.linalg.norm(axis)
        axis_length = np.linalg.norm(axis)

        if projection_length < 0 or projection_length > axis_length:
            return False

    return True

def prune_bild_data(bild_data, p1, p2, width, height):
    obb_corners = generate_OBB(p1, p2, width, height)
    return [point for point in bild_data if is_inside_OBB(point, obb_corners)]

def get_opposite_corners(corners):
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    return corners[0], corners[-1]#min_corner.tolist(), max_corner.tolist()

def generate_box_bild(o, corners):
    faces = [
        [corners[0], corners[1], corners[5], corners[4]],  # Front face
        [corners[2], corners[3], corners[7], corners[6]],  # Back face
        [corners[0], corners[2], corners[6], corners[4]],  # Bottom face
        [corners[1], corners[3], corners[7], corners[5]],  # Top face
        [corners[0], corners[2], corners[3], corners[1]],  # Left face
        [corners[4], corners[6], corners[7], corners[5]],  # Right face
    ]
    with open(o, 'w') as out:
        out.write('.transparency 0.5\n')
        out.write('.color %.5f %.5f %.5f\n' % (0.00, 1.00, 0.00))
        for face in faces:
            flat_face = [coord for vertex in face for coord in vertex]
            out.write(".polygon " + " ".join("%.5f" % coord for coord in flat_face) + "\n")
    out.close()

def nearest_neighbor_order(points):
    centroid = np.mean(points, axis=0)
    starting_point = max(points, key=lambda point: np.linalg.norm(np.array(point) - centroid))
    ordered = [starting_point]
    remaining_points = [tuple(p) for p in points]
    remaining_points.remove(tuple(starting_point))
    current_point = starting_point
    while remaining_points:
        next_point = min(remaining_points, key=lambda point: np.linalg.norm(np.array(point) - np.array(current_point)))
        ordered.append(next_point)
        remaining_points.remove(next_point)
        current_point = next_point
        
    return ordered

def interface_length(spline1, spline2, threshold_distance):
    spline1 = np.array(spline1)[::100]
    spline2 = np.array(spline2)[::100]
    contact_points = []
    for p1 in spline1:
        distances = np.linalg.norm(spline2 - p1, axis=1)
        if np.min(distances) < threshold_distance:
            contact_points.append(p1)
    if not contact_points:
        return 0
    contact_points = np.array(contact_points)
    diffs = np.diff(contact_points, axis=0)
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

import os
################################################################################
mrc_file_name_base = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/semSeg_actin/'
cmm_file_name_base = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/automated_fil_identification_stable_cores_forFig/'
bild_file_name_base = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/'
energyScores_file_name = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/filament_identification/allEnergyScores.txt'
#tomo_name_list = ['ts044', 'ts045', 'ts053', 'ts058', 'ts059', 'ts062', 'ts064', 'ts065', 'ts072', 'ts074']
#tomo_name_list = ['ts074']#, 'ts045', 'ts053', 'ts058', 'ts059', 'ts062', 'ts064', 'ts065', 'ts072', 'ts074']
#tomo_name_list = ['ts044_A', 'ts059_A','ts059_B','ts059_C','ts059_D']
tomo_name_list = ['ts059_B']#, 'ts045_A', 'ts045_B', 'ts045_C', 'ts045_D', 'ts053_A', 'ts053_B', 'ts058_A', 'ts058_B','ts058_C','ts059_A','ts059_B','ts059_C','ts059_D','ts062_A','ts064_A','ts064_B','ts064_C','ts065_A','ts072_A','ts072_C','ts074_A','ts074_B','ts074_C']
for i in range(0, len(tomo_name_list)):
    file_name = tomo_name_list[i]
    print('Doing traces on ' + file_name)
    # Load in traces
    print('Loading in manual traces...')
    cmm_file_names = sorted(glob.glob(cmm_file_name_base+tomo_name_list[i]+'/*.cmm'))
    os.makedirs(cmm_file_name_base+tomo_name_list[i]+'/graph_theory', exist_ok=True)
    os.makedirs(cmm_file_name_base+tomo_name_list[i]+'/paired_bilds', exist_ok=True)
    manual_traces = []
    for j in range(0, len(cmm_file_names)):
        manual_traces.append(load_cmm_data(cmm_file_names[j]).astype(float))
    print('Manual traces loaded.')
    for j in range(0, len(manual_traces)):
        manual_traces[j] = np.asarray(nearest_neighbor_order(manual_traces[j]))
    # Load in image data; denoised tomogram
    print('Loading in denoised map...')
    with mrcfile.open(mrc_file_name_base + tomo_name_list[i][:-2]+'.st_rec_channel0.mrc') as mrc:
        denoised_map = mrc.data.astype('float32')
    print('Denoised map loaded.')

    DEFAULT_S = 1 #0.5 works and gives pretty good results
    curve_point_holder = []
    for j in range(0, len(manual_traces)):
    #    mrc_output_path = cmm_file_name_base + tomo_name_list[i]+'/actin_spline_bilds/'+'%s_spheres_optimized.mrc'%str(cmm_file_names[j].split('/')[-1][:-4])
    #    temp = compute_spline_and_volume(manual_traces[j]/3.0, denoised_map, mrc_output_path)
        temp = compute_spline_only(manual_traces[j]/3.0, 100)
        curve_point_holder.append(temp*3)

    # Load BILD file
    bild_file_name = bild_file_name_base+tomo_name_list[i][:-2]+'.bild'
    bild_data = load_bild_data(bild_file_name)
    
    # Eliminate bild points outside of the manually picked bounding box
    #WIDTH = 450
    #p1 = (437.841, 4041.921, 613.017)
    #p2 = (1707.543, 4848.624, 312.255)
    #print('Bounding box length: ' + str(np.linalg.norm(np.array(p2)-np.array(p1))))
    #bild_data_pruned = prune_bild_data(bild_data, p1, p2, 1000, WIDTH)
    #box_biild_output_name = cmm_file_name_base + tomo_name_list[i][:-2]+'/paired_bilds/pruning_box.bild'
    #generate_box_bild(box_biild_output_name, generate_OBB(p1,p2, 1000, WIDTH))
    bild_data_pruned = bild_data #TODO
    cutoff_distance = 30
    groups, rejected_points = group_points_by_splines(bild_data_pruned, curve_point_holder, cutoff_distance)
    for splines, grouped_points in groups.items():
        print(f"Splines: {list(splines)}, Points count: {len(grouped_points)}")
    print(f"Rejected points count: {len(rejected_points)}")

    retained_points = [point for _, group in groups.items() for point in group]
    
    # Apply ordering for each group
    ordered_groups = {}
    for splines, points in groups.items():
        spline1 = curve_point_holder[list(splines)[0]]
        spline2 = curve_point_holder[list(splines)[1]]
        
        ordered_groups[splines] = order_points_along_path(points, spline1, spline2)

    # If you want to visualize or save the ordered points, you can do so here
    for splines, ordered_points in ordered_groups.items():
        print(f"Splines: {splines}, Ordered Points: {ordered_points.shape}")
        #bild_output_name = cmm_file_name_base + bild_file_names[0].split('.')[0]+'/actin_spline_bilds/'+bild_file_names[0].split('.')[0]+'_actinSpline_'+str(list(splines)[0])+'_'+str(list(splines)[1])+'.bild'
        #generate_bild(bild_output_name, ordered_points, np.random.rand(3))

    energyScores = np.loadtxt(energyScores_file_name,dtype=str)
    energyValues_thisTomo = energyScores[energyScores[:,2] == tomo_name_list[i][:-2]]

    # Assuming these are the column indices in filtered_data where the 3D points and energy scores are located
    x_col, y_col, z_col = 3,4,5 
    x_check =  energyValues_thisTomo[:, x_col].astype(float)
    y_check =  energyValues_thisTomo[:, y_col].astype(float)
    z_check =  energyValues_thisTomo[:, z_col].astype(float)
    energy_col = 1  
    my_point_holder_plus_energy = []
    spline_keys = []
    TOLERANCE = 1e-2
    for splines, point_set in tqdm(ordered_groups.items()):
        # Compute spline contact length
        spline1 = curve_point_holder[list(splines)[0]]
        spline2 = curve_point_holder[list(splines)[1]]
        this_interface_length = interface_length(spline1, spline2, 2*cutoff_distance) #TODO
        extended_set = []  # This will hold points with energy scores
        for point in point_set:
            # Instead of checking for string equality, use a numerical close-enough check
            mask = (
                np.isclose(x_check, point[0], atol=TOLERANCE) &
                np.isclose(y_check, point[1], atol=TOLERANCE) &
                np.isclose(z_check, point[2], atol=TOLERANCE)
            )
            matched_row = energyValues_thisTomo[mask]
            
            # If there's a match, fetch the energy score and append it to the point
            if matched_row.size > 0:
                energy_score = float(matched_row[0, energy_col])  # Convert the energy score to float
                extended_set.append(np.append(point, (energy_score, this_interface_length)))
            else:
                extended_set.append(np.append(point, np.nan, np.nan))
                    
        my_point_holder_plus_energy.append(np.asarray(extended_set))
        spline_keys.append(splines)
    
    # Create the colormap
    colors = [(1, 1, 1), (1, 0, 0)]  # This corresponds to [white, red]
    n_bins = 1000  # Discretizes the interpolation into bins
    import matplotlib.colors as mcolors
    cmap_name = 'custom_white_to_red'
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    norm = plt.Normalize(vmin=1.50, vmax=3.0)# 2 and 2.4 for each fascin
    # Create a colormap instance
    #cmap = plt.cm.get_cmap('bwr')
    print(np.asarray(cmap(0.0))*255.0, np.asarray(cmap(0.25))*255.0, np.asarray(cmap(0.5))*255.0, np.asarray(cmap(0.75))*255.0, np.asarray(cmap(3.0))*255.0)
    #cmap = plt.cm.get_cmap('bwr')
    for j in range(0, len(my_point_holder_plus_energy)):
        num_fascin_bridges = len(my_point_holder_plus_energy[j][:,3])
        computed_interface_length = my_point_holder_plus_energy[j][0,4]/1000
        avg_energy_score = np.mean(my_point_holder_plus_energy[j][:,3])
        print('The total number of fascin bridges is: ' + str(num_fascin_bridges))
        print('The computed interface length is: ' + str(computed_interface_length))
        print('The number of fascins per unit length is: ' + str(num_fascin_bridges/computed_interface_length))
        print('The average energy score is: ' + str(avg_energy_score))
        # IF you want to plot each fascin a different color:
        this_pairs_energy_score = avg_energy_score
        # ELSE if you want to color everything the average interface score
        #this_pairs_energy_score = num_fascin_bridges/computed_interface_length / (1+avg_energy_score)
        print(this_pairs_energy_score)
        this_pairs_color = cmap(norm(this_pairs_energy_score))
        #this_pairs_color = cmap(norm(my_point_holder_plus_energy[j][:,3]))
        bild_output_name = cmm_file_name_base + tomo_name_list[i]+'/paired_bilds/'+tomo_name_list[i][:-2]+'_paired_fils_energy_'+str(j).zfill(3)+'.bild'
        # IF you want to plot each fascin as a different color
        generate_bild(bild_output_name, my_point_holder_plus_energy[j],  cmap(norm(my_point_holder_plus_energy[j][:,3])), avg_energy_score)
        # ELSE if you want to color everything the average interface score
        #generate_bild(bild_output_name, my_point_holder_plus_energy[j],  this_pairs_color, this_pairs_energy_score)

    # Time for graph theory:
    G = nx.Graph()
    unique_nodes = list(set({value for spline in spline_keys for value in spline}))
    G.add_nodes_from(unique_nodes)
    for node in unique_nodes:
        G.nodes[node]['filament_path'] = cmm_file_names[node]
    for j, (start, end) in enumerate(spline_keys):
        num_fascin_bridges = len(my_point_holder_plus_energy[j][:,3])
        computed_interface_length = my_point_holder_plus_energy[j][0,4]/1000
        avg_energy_score = np.mean(my_point_holder_plus_energy[j][:,3])
        this_pairs_energy_score = num_fascin_bridges/computed_interface_length / (1+avg_energy_score)
        weight = np.exp(-1*(this_pairs_energy_score))
        G.add_edge(start, end, weight=weight)

    multiG = nx.MultiGraph()
    for j, (start, end) in enumerate(spline_keys):
        fascin_bridge_scores = my_point_holder_plus_energy[j][:, 3]
        weights = np.exp(-1 * (fascin_bridge_scores))
        for k in range(0, len(weights)):
            multiG.add_edge(start, end, weight=weights[k])    # Extract communities and compute modularity
    
    def compute_modularity(G, communities):
        m = G.number_of_edges()
        Q = 0
        for community in communities:
            in_edges = G.subgraph(community).number_of_edges()
            out_edges = sum(G.degree(n) for n in community)
            Q += (in_edges / m) - (out_edges / (2 * m))**2
        return Q

    comp = list(girvan_newman(G))  # Convert the generator to a list
    modularities = [compute_modularity(G, [list(c) for c in communities]) for communities in comp]

    # Get optimal number of communities based on modularity
    optimal_communities = max(zip(modularities, comp), key=lambda x: x[0])[1]
    #optimal_communities = comp[4]

    # Plot modularity vs number of communities
    plt.figure()
    plt.plot(range(1, len(modularities) + 1), modularities, 'o-', color='b')
    plt.axvline(modularities.index(max(modularities)) + 1, color='r', linestyle='--')
    plt.xlabel('Number of Communities')
    plt.ylabel('Modularity')
    plt.title('Modularity vs Number of Communities')
    graph_img_name = cmm_file_name_base + tomo_name_list[i]+'/graph_theory/'
    plt.savefig(graph_img_name+'graphD.png', format='png')
    plt.clf()

    # Visualize optimal communities
    pos = nx.spring_layout(G, iterations=1000)  # Layout for nodes

    # Dynamic color palette based on number of communities
    colors = [plt.cm.jet(x) for x in np.linspace(0, 1, len(optimal_communities))]

    plt.figure(figsize=(10, 7))
    for idx, community in enumerate(optimal_communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=colors[idx], node_size=100)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title("Optimal Communities in Filament Graph")
    plt.savefig(graph_img_name + 'graph.png', format='png')
    plt.clf(); plt.close()

    def draw_multiedges(G, pos, edgelist=None, width=1.0):
        if edgelist is None:
            edgelist = G.edges()
        
        for edge in edgelist:
            u, v, key = edge
            # Calculate the curve factor based on the key
            # You can adjust the factor for a more pronounced curve
            curve_factor = key * 0.2  
            
            # Calculate the middle point of the straight line
            point1 = pos[u]
            point2 = pos[v]
            mid_point = [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]
            
            # Add a curve by adjusting the middle point
            mid_point[1] += curve_factor
            
            # Draw the quadratic bezier curve
            bezier_line = [point1, mid_point, point2]
            quadratic_bezier(bezier_line, width=width)

    def quadratic_bezier(points, width=1.0):
        t = np.linspace(0, 1, 100)
        bezier = np.outer((1 - t) ** 2, points[0]) + np.outer(2 * (1 - t) * t, points[1]) + np.outer(t ** 2, points[2])
        plt.plot(bezier[:, 0], bezier[:, 1], lw=width)

    pos = nx.spring_layout(multiG) 
    nx.draw_networkx_nodes(multiG, pos)
    for u, v, key, attributes in multiG.edges(keys=True, data=True):
        weight = attributes['weight']
        draw_multiedges(multiG, pos, edgelist=[(u, v, key)], width=weight)

    nx.draw_networkx_labels(multiG, pos)
    plt.title("Multigraph Visualization")
    plt.savefig(graph_img_name + 'graph_multi.png', format='png')
    plt.clf(); plt.close()
    
    # Save the graph, G
    with open(graph_img_name+'graph.pkl', 'wb') as f:
        pickle.dump(G,f)
    
    # Save the graph, G
    with open(graph_img_name+'multigraph.pkl', 'wb') as f:
        pickle.dump(multiG,f)
    

print('Hi')






