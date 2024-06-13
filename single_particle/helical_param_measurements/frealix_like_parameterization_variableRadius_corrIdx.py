#!/home/greg/.conda/envs/matt_TF/bin/python
# Import python files
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from prody import *
from scipy.interpolate import CubicSpline
import string; import sys
################################################################################
from scipy.interpolate import UnivariateSpline
def define_axis_spline_curve(x,y,z,res=0.1):
	cs_x = UnivariateSpline(np.arange(0,len(x)), x, s=len(x)*0.001)
	x_spline = cs_x(np.arange(-1, len(x), res))
	cs_y = UnivariateSpline(np.arange(0,len(y)), y, s=len(y)*0.001)
	y_spline = cs_y(np.arange(-1, len(y), res))
	cs_z = UnivariateSpline(np.arange(0,len(z)), z, s=len(z)*0.001)
	z_spline = cs_z(np.arange(-1, len(z), res))
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

def deriv_axis_spline_curve(x,y,z,res=0.1,order=1):
	cs_x = UnivariateSpline(np.arange(0,len(x)), x, s=len(x)*0.001)
	x_spline = cs_x(np.arange(-1, len(x), res),order)
	cs_y = UnivariateSpline(np.arange(0,len(y)), y, s=len(y)*0.001)
	y_spline = cs_y(np.arange(-1, len(y), res),order)
	cs_z = UnivariateSpline(np.arange(0,len(z)), z, s=len(z)*0.001)
	z_spline = cs_z(np.arange(-1, len(z), res),order)
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

def get_coord_of_min_dist(pt, axis, init_radius):
	pt_repeat = np.repeat(np.expand_dims(pt, axis=0), len(axis), axis=0)
	dists = np.linalg.norm(pt_repeat - axis, axis=1)
	min_dist_arg = np.argmin(dists)
	vect_to_center = axis[min_dist_arg] - pt
	vect_to_center = vect_to_center / np.linalg.norm(vect_to_center) * init_radius#15.76719916835962
	discrepancy = np.linalg.norm(vect_to_center+pt - axis[min_dist_arg])
	if(init_radius <= dists[min_dist_arg]):
		updated_radius = init_radius+discrepancy-1.0
	else:
		updated_radius = init_radius-discrepancy-1.0
	
	#vect_to_center = vect_to_center / np.linalg.norm(vect_to_center) * updated_radius#15.76719916835962
	return vect_to_center, updated_radius

def measure_variable_radius(pt, axis):
	pt_repeat = np.repeat(np.expand_dims(pt, axis=0), len(axis), axis=0)
	dists = np.linalg.norm(pt_repeat - axis, axis=1)
	min_dist_arg = np.argmin(dists)
	return dists[min_dist_arg]

# get the argument along the axis spline that corresponds to the subunit
def get_h_of_min_dist(pt, axis):
	pt_repeat = np.repeat(np.expand_dims(pt, axis=0), len(axis), axis=0)
	dists = np.linalg.norm(pt_repeat - axis, axis=1)
	min_dist_arg = np.argmin(dists)
	return min_dist_arg

# Now, try and get phis
def rotate_yaw(alpha, vector):
	R_yaw = np.array([[np.cos(alpha), -1.0*np.sin(alpha), 0],[np.sin(alpha), np.cos(alpha), 0],[0,0,1]])
	rot_vect = np.matmul(R_yaw, vector)
	return rot_vect

# Get phis by finding optimal angle based on distance to centroid of rotated vector
def get_phis_optAng(axis_pt, true_pos, rot_matrix):
	vect_to_rotate_to = axis_pt - true_pos#np.matmul(rot_matrix.T, true_pos - axis_pt)
	tang_vect = np.array([15.0,0,0])#np.array([2,0,0])#np.matmul(rot_matrix, np.array([15.76719916835962,0,0]))
	best_dist = 10000
	samp_rate = np.radians(90)
	best_alpha = np.radians(0); 
	it_cnt = 0
	# gradient descent to get best phi
	while(best_dist > 0.000001 and samp_rate > np.radians(0.000005) and it_cnt < 100000):
		alpha = best_alpha
		decrease_sample_rate = False
		for i in range(-1,2):
			rotated_vect = rotate_yaw(alpha+i*samp_rate, tang_vect)
			new_vect = np.matmul(rot_matrix, rotated_vect) #+ axis_pt
			temp_dist = np.linalg.norm(new_vect - vect_to_rotate_to)
			if(temp_dist <= best_dist):
				best_dist = temp_dist
				best_alpha = alpha+i*samp_rate
				if(i == 0):
					decrease_sample_rate = True
		if(decrease_sample_rate):
			samp_rate = samp_rate / 2.0
		it_cnt = it_cnt + 1
	print(best_dist, np.linalg.norm(new_vect), np.linalg.norm(vect_to_rotate_to))
	print(vect_to_rotate_to, new_vect)
	return best_alpha, best_dist

def get_R_from_a_to_b(a, b):
	v = np.cross(a,b)
	skew_symm_cp = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
	a_b_dot = np.dot(a,b)
	R = np.eye(3) + skew_symm_cp + np.matmul(skew_symm_cp, skew_symm_cp) * 1.0/(1.0+a_b_dot)
	return R

################################################################################
def matt_parsePDB(file_name):
	with open(file_name, 'r') as f:
		lines = f.readlines()
	
	chains, coords = [] ,[]
	chainNum = 0
	for i in range(0, len(lines)):
		if(lines[i][:3]  == 'TER'):
			chainNum = chainNum + 1
		if(lines[i][:4] == 'ATOM' and lines[i][13:15] == 'CA'):
			chain,x,y,z = lines[i][21] + str(chainNum), float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])
			chains.append(chain); coords.append((x,y,z))
			#print lines[i][21] + str(chainNum),
	
	chains = np.asarray(chains)
	coords = np.asarray(coords)
	chain_idxs = list(set(chains))
	coords_per_chain = []
	for i in range(0, len(chain_idxs)):
		coords_per_chain.append(coords[np.argwhere(chains == chain_idxs[i])][:,0])
	
	return np.asarray(coords_per_chain)

################################################################################
################################################################################
# Load in PDB file and get chains
# Load in PDB file and get chains
frame_index = './actin_side4_IndividualFit_stitch.pdb'
param_file_name = './measured_params/'+frame_index[:-4]+'.csv'
file_name = frame_index#'./stitched_fit_pdbs_3dva/stitched_masterSquigJ38_'+frame_index+'_final_C.pdb'
def load_coords(file_name):
	# make each helix into a [num_chains x num_atoms_per_actin x 3] array
	coords = matt_parsePDB(file_name)
	centroids = np.average(coords, axis=1)
	
	# sort centroids by z
	z_index = np.argsort(centroids[:,2])[::-1]
	centroids = centroids[z_index]
	return centroids

#centroids_top = load_coords(top_file_name)
#centroids_mid = load_coords(mid_file_name)
#centroids_bot = load_coords(bot_file_name)
centroids =load_coords(file_name)# np.concatenate((centroids_bot, centroids_mid, centroids_top))

centroids = centroids - np.average(centroids, axis=0)
#centroids = centroids[2:-2]
avgs = (centroids[:-1] + centroids[1:]) / 2.0

print(len(avgs))
print(len(centroids))
plt.plot(np.arange(0,43)[::2][:-1], np.linalg.norm(centroids[::2][:-1] - centroids[::2][1:], axis=-1), marker='o')
#plt.plot(np.arange(0,42)[1::2][:-2], np.linalg.norm(centroids[1::2][:-1] - centroids[1::2][1:], axis=-1), marker='o')
plt.show()

# plot the centroids in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(avgs[:,0], avgs[:,1], avgs[:,2], c='red')
_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()



################################################################################
# Now that the data is loaded in, define an initial axis
radius_deviations = np.ones((centroids.shape[0]))*12.0
orig_axis = define_axis_spline_curve(avgs[:,0], avgs[:,1], avgs[:,2])
center_pointers = np.zeros((centroids.shape))
for i in range(0, len(centroids)):
	center_pointers[i], radius_deviations[i] = get_coord_of_min_dist(centroids[i], orig_axis, radius_deviations[i])

new_avgs = ((center_pointers+centroids)[:-1] + avgs + (center_pointers+centroids)[1:])/3.0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(orig_axis[:,0], orig_axis[:,1], orig_axis[:,2], c='red')
_=ax.scatter(avgs[:,0], avgs[:,1], avgs[:,2], c='blue', s=100)
_=ax.scatter(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], c='orange', s=100)
for i in range(0, len(center_pointers)):
	a=np.array([centroids[i], center_pointers[i] + centroids[i]])
	_=plt.plot(a[:,0],a[:,1],a[:,2], c='purple')

_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()

################################################################################
# Iteratively update the central axis estimate
radius_deviations = np.ones((centroids.shape[0]))*10.0
orig_axis = define_axis_spline_curve(avgs[:,0], avgs[:,1], avgs[:,2])
center_pointers = np.zeros((centroids.shape))
for i in range(0, len(centroids)):
	center_pointers[i], radius_deviations[i] = get_coord_of_min_dist(centroids[i], orig_axis, radius_deviations[i])

new_avgs = ((center_pointers+centroids)[:-1] + avgs + (center_pointers+centroids)[1:])/3.0

prev_avgs = new_avgs.copy()
radius_deviations = np.ones((centroids.shape[0]))*10.0
for i in tqdm(range(0, 500)):
	temp_axis = define_axis_spline_curve(prev_avgs[:,0], prev_avgs[:,1], prev_avgs[:,2], 0.01)
	center_pointers = np.zeros((centroids.shape))
	for j in range(0, len(centroids)):
		center_pointers[j], radius_deviations[j] = get_coord_of_min_dist(centroids[j], temp_axis, radius_deviations[j])
	radius_deviations[0] = np.mean(radius_deviations[1:-1])
	radius_deviations[-1] = np.mean(radius_deviations[1:-1])
	new_avgs = ((center_pointers+centroids)[:-1] + prev_avgs + (center_pointers+centroids)[1:])/3.0
	prev_avgs = new_avgs.copy()
	cs_rd = UnivariateSpline(np.arange(0,len(radius_deviations)), radius_deviations, s=len(radius_deviations)*0.1) # originally 0.2
	radius_deviations = cs_rd(np.arange(0, len(radius_deviations), 1))
	if(i%249 ==0):
		plt.plot(radius_deviations, marker='o')
		plt.show()


# Final axis, for viewing. When doing more calculations, sample more finely
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.05)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(final_axis[:,0], final_axis[:,1], final_axis[:,2], c='red')
_=ax.scatter(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], c='orange', s=100)
for i in range(0, len(center_pointers)):
	a=np.array([centroids[i], center_pointers[i] + centroids[i]])
	_=plt.plot(a[:,0],a[:,1],a[:,2], c='purple')

_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()

# Plot, x(t), y(t), z(t)
fig, ax = plt.subplots(1,3)
ax[0].plot(final_axis[:,0])
ax[1].plot(final_axis[:,1])
ax[2].plot(final_axis[:,2])
plt.show()

################################################################################
# Get final radius changes
# get distance between steps along central axis curve
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
time_steps = np.linalg.norm(final_axis[:-1] - final_axis[1:], axis=1)

radii = []
for i in range(0, len(centroids)):
	radii.append(measure_variable_radius(centroids[i], final_axis))

radii = np.asarray(radii)
cs_rd = UnivariateSpline(np.arange(0,len(radii)), radii, s=len(radii)*0.1) # originally 0.2
radii = cs_rd(np.arange(0, len(radii), 1))
plt.plot(np.arange(1,len(radii),1)[::2], radii[1:][::2], marker = 'o')
plt.plot(np.arange(1,len(radii),1)[1::2],radii[1:][1::2], marker = 'o')
plt.xlim(16,28)
plt.show()

plt.plot(radii,marker='o')
plt.xlim(16,28)
plt.show()

################################################################################
# get distance between steps along central axis curve
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
time_steps = np.linalg.norm(final_axis[:-1] - final_axis[1:], axis=1)

# Now that we have the central axis as final_axis, get rise and twist
# Plot h'(t) 
hs = []
for i in range(0, len(centroids)):
	hs.append(get_h_of_min_dist(centroids[i], final_axis))

hs = np.asarray(hs)

cs = CubicSpline(np.arange(0,len(hs)), hs, bc_type='natural')
h_of_t = cs(np.arange(0, len(hs), 0.001))[:-1] * time_steps
h_of_t_prime = cs(np.arange(0, len(hs), 0.001), 1)[:-1] * time_steps 
plt.plot(h_of_t_prime) # resize b/c pixel size is 1.03A/px
plt.scatter(hs, h_of_t_prime[hs])
plt.ylim(26.5,29.5)
plt.show()


#plt.plot(h_of_t_prime) # resize b/c pixel size is 1.03A/px
plt.plot(hs[::2], h_of_t_prime[hs][::2], marker='o')
plt.plot(hs[1::2], h_of_t_prime[hs][1::2], marker='o')
plt.ylim(24,32)
plt.xlim(16000,28000)
plt.show()


################################################################################
################################################################################
# define r'(t)
final_axis_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
# define r"(t) for later
final_axis_second_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001, 2)
# define T(t) = r'(t)/||r'(t)||
r_prime_norms = np.repeat(np.expand_dims(np.linalg.norm(final_axis_deriv, axis=-1), axis=-1), 3, axis=-1)
T_of_t = final_axis_deriv / r_prime_norms

rots = []
for i in range(0, len(T_of_t)):
	rots.append(get_R_from_a_to_b(np.array([0,0,-1]), T_of_t[i]))

rots = np.asarray(rots)

phis = []; dists = []
for i in range(0, len(centroids)):
	phis.append(get_phis_optAng(final_axis[hs[i]], centroids[i],rots[i*1000])[0])
	dists.append(get_phis_optAng(final_axis[hs[i]], centroids[i],rots[i*1000])[1])

phis = np.asarray(phis)
delta_phis = []
for i in range(0, len(phis)-1):
	delta = (np.degrees(phis[i+1]) - np.degrees(phis[i]))
	if(delta < 0):
		delta = delta +360
	delta_phis.append(-1*delta)

dists = np.asarray(dists)
delta_phis = np.asarray(delta_phis)

plt.plot(np.asarray(delta_phis), marker='o')
plt.ylim(-190,-145)
plt.show()

deltas = delta_phis.copy()
plt.plot(np.arange(1,len(deltas),1)[::2], deltas[1:][::2], marker = 'o')
plt.plot(np.arange(1,len(deltas),1)[1::2],deltas[1:][1::2], marker = 'o')
plt.ylim(-200,-140)
plt.xlim(16,28)
plt.show()



################################################################################
twists = np.asarray(delta_phis)[1:]
rises = np.asarray(h_of_t_prime[hs][1:-1])
vals_to_save = np.concatenate((np.expand_dims(twists, axis=-1), np.expand_dims(rises, axis=-1)), axis=-1)
np.savetxt(param_file_name, vals_to_save, delimiter=',')
np.savetxt(param_file_name[:-4] + '_centralAxisCoords.txt', new_avgs, delimiter=',')
np.savetxt(param_file_name[:-4] + '_subunitCentroids.txt', centroids, delimiter=',')
#sys.exit()




# Save twist and rise values to external file
import pandas as pd
twists = np.asarray(deltas)[16:28]
rises = np.asarray(h_of_t_prime[hs][16:28])
long_pitch_twists = 360+twists[1:] + twists[:-1]
long_pitch_rises  = rises[1:] + rises[:-1]
isds = np.linalg.norm(centroids[16:28][:-2] - centroids[16:28][2:], axis=-1)
central_axis_sampling = final_axis.copy()
centroids_subset = centroids[16:28]
radii_subset = radii[16:28]

if(len(twists) %2 != 0):
	twists = np.append(twists, '')
df_twist = pd.DataFrame(np.asarray([np.repeat(twists[::2],2), np.repeat(twists[1::2],2)]).T, columns=['twist pf2', 'twist pf1'])
df_twist['twist pf2'].iloc[1::2] = ''
df_twist['twist pf1'].iloc[::2] = ''

if(len(rises) %2 != 0):
	rises = np.append(rises, '')
df_rise = pd.DataFrame(np.asarray([np.repeat(rises[::2],2), np.repeat(rises[1::2],2)]).T, columns=['rise pf2', 'rise pf1'])
df_rise['rise pf2'].iloc[1::2] = ''
df_rise['rise pf1'].iloc[::2] = ''

#df_longTwist = pd.DataFrame(np.asarray([np.repeat(long_pitch_twists[::2],2), np.repeat(long_pitch_twists[1::2],2)]).T, columns=['long-pitch twist pf2', 'long-pitch twist pf1'])
#df_longTwist['long-pitch twist pf2'].iloc[1::2] = ''
#df_longTwist['long-pitch twist pf1'].iloc[::2] = ''

#df_longRise = pd.DataFrame(np.asarray([np.repeat(long_pitch_rises[::2],2), np.repeat(long_pitch_rises[1::2],2)]).T, columns=['long-pitch rise pf2', 'long-pitch rise pf1'])
#df_longRise['long-pitch rise pf2'].iloc[1::2] = ''
#df_longRise['long-pitch rise pf1'].iloc[::2] = ''

if(len(isds) %2 !=0):
	isds = np.append(isds, '')
df_isd = pd.DataFrame(np.asarray([np.repeat(rises[::2],2), np.repeat(rises[1::2],2)]).T, columns=['Inter-subunit dist. pf2', 'Inter-subunit dist. pf1'])
df_isd['Inter-subunit dist. pf2'].iloc[1::2] = ''
df_isd['Inter-subunit dist. pf1'].iloc[::2] = ''

df_centerSpline = pd.DataFrame(central_axis_sampling, columns=['x','y','z'])
df_centroids = pd.DataFrame(centroids, columns=['x','y','z'])
df_centroids_subset = pd.DataFrame(centroids_subset, columns=['x','y','z'])
df_radius = pd.DataFrame(radii_subset, columns=['radius'])

df_twist.to_excel(param_file_name[:-4] + '_twist.xlsx')
df_rise.to_excel(param_file_name[:-4] + '_rise.xlsx')
#df_longTwist.to_excel(param_file_name[:-4] + '_longPitchTwist.xlsx')
#df_longRise.to_excel(param_file_name[:-4] + '_longPitchRise.xlsx')
df_isd.to_excel(param_file_name[:-4] + '_intersubunitDistance.xlsx')
df_centerSpline.to_excel(param_file_name[:-4] + '_centerSplineSampling.xlsx')
df_centroids.to_excel(param_file_name[:-4] + '_centroids.xlsx')
df_centroids_subset.to_excel(param_file_name[:-4] + '_centroids_subset.xlsx')
df_radius.to_csv(param_file_name[:-4] + '_changingRadius_subset.csv', sep=' ')
sys.exit()

################################################################################
################################################################################
################################################################################
# Now, get curvature of the central axis
r_prime = final_axis_deriv.copy()
r_double_prime = final_axis_second_deriv.copy()
k = np.linalg.norm(np.cross(r_prime, r_double_prime), axis=-1) / (np.linalg.norm(r_prime, axis=-1)**3)


# The fun stuff: plot h(t), phi(t), K(t) as fxns of each other
curv = k[:-1][hs][1:-1]
rise = h_of_t_prime[hs][1:-1]
twist = np.degrees(phi_of_t_prime[hs][1:-1])

fig, ax = plt.subplots(1,3)
ax[0].scatter(rise, twist)
ax[1].scatter(1.0/curv, twist)
ax[2].scatter(1.0/curv,  rise)
plt.tight_layout()
plt.show()

plt.plot(1.0/k)
plt.ylim(-5000,10000)
plt.show()

################################################################################
# Compute discrepancy between extended line and cylindrical radius
from scipy.interpolate import UnivariateSpline
def define_axis_spline_curve(x,y,z,res=0.1):
	cs_x = CubicSpline(np.arange(0,len(x)), x, bc_type='natural')
	x_spline = cs_x(np.arange(-1, len(x), res))
	cs_y = CubicSpline(np.arange(0,len(y)), y, bc_type='natural')
	y_spline = cs_y(np.arange(-1, len(y), res))
	cs_z = CubicSpline(np.arange(0,len(z)), z, bc_type='natural')
	z_spline = cs_z(np.arange(-1, len(z), res))
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

def deriv_axis_spline_curve(x,y,z,res=0.1, order=1):
	cs_x = CubicSpline(np.arange(0,len(x)), x, bc_type='natural')
	x_spline = cs_x(np.arange(-1, len(x), res),order)
	cs_y = CubicSpline(np.arange(0,len(y)), y, bc_type='natural')
	y_spline = cs_y(np.arange(-1, len(y), res),order)
	cs_z = CubicSpline(np.arange(0,len(z)), z, bc_type='natural')
	z_spline = cs_z(np.arange(-1, len(z), res),order)
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

def get_coord_of_min_dist(pt, axis):
	pt_repeat = np.repeat(np.expand_dims(pt, axis=0), len(axis), axis=0)
	dists = np.linalg.norm(pt_repeat - axis, axis=1)
	min_dist_arg = np.argmin(dists)
	vect_to_center = axis[min_dist_arg] - pt
	vect_to_center = vect_to_center / np.linalg.norm(vect_to_center) * 10#15.76719916835962
	discrepancy = np.linalg.norm(vect_to_center+pt - axis[min_dist_arg])
	return vect_to_center, discrepancy


radius_deviations = np.zeros((centroids.shape[0]))
orig_axis = define_axis_spline_curve(avgs[:,0], avgs[:,1], avgs[:,2])
center_pointers = np.zeros((centroids.shape))
for i in range(0, len(centroids)):
	center_pointers[i], radius_deviations[i] = get_coord_of_min_dist(centroids[i], orig_axis)

new_avgs = ((center_pointers+centroids)[:-1] + avgs + (center_pointers+centroids)[1:])/3.0

# Iteratively update the central axis estimate
prev_avgs = new_avgs.copy()
for i in tqdm(range(0, 500)):
	temp_axis = define_axis_spline_curve(prev_avgs[:,0], prev_avgs[:,1], prev_avgs[:,2], 0.01)
	center_pointers = np.zeros((centroids.shape))
	radius_deviations = np.zeros((centroids.shape[0]))
	for j in range(0, len(centroids)):
		center_pointers[j], radius_deviations[j] = get_coord_of_min_dist(centroids[j], temp_axis)
	new_avgs = ((center_pointers+centroids)[:-1] + prev_avgs + (center_pointers+centroids)[1:])/3.0
	prev_avgs = new_avgs.copy()

# Final axis, for viewing. When doing more calculations, sample more finely
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.05)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
_=ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c='green', alpha=0.2, s=300)
_=ax.scatter(final_axis[:,0], final_axis[:,1], final_axis[:,2], c='red')
_=ax.scatter(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], c='orange', s=100)
for i in range(0, len(center_pointers)):
	a=np.array([centroids[i], center_pointers[i] + centroids[i]])
	_=plt.plot(a[:,0],a[:,1],a[:,2], c='purple')

_=ax.set_xlim(-200,200); _=ax.set_ylim(-200,200); _=ax.set_zlim(-200,200)
plt.show()


plt.plot(radius_deviations, marker='o')
plt.show()

np.mean(radius_deviations[18:-18])


# Do new stuff
def define_axis_spline_curve(x,y,z,res=0.1):
	cs_x = UnivariateSpline(np.arange(0,len(x)), x, s=len(x)*0.1)
	x_spline = cs_x(np.arange(-1, len(x), res))
	cs_y = UnivariateSpline(np.arange(0,len(y)), y, s=len(y)*0.1)
	y_spline = cs_y(np.arange(-1, len(y), res))
	cs_z = UnivariateSpline(np.arange(0,len(z)), z, s=len(z)*0.1)
	z_spline = cs_z(np.arange(-1, len(z), res))
	central_axis = np.stack((x_spline, y_spline, z_spline),axis=-1)
	return central_axis

final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.05)
radius_deviations = np.zeros((centroids.shape[0]))
center_pointers = np.zeros((centroids.shape))
final_axis = define_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.05)
for j in range(0, len(centroids)):
	center_pointers[j], radius_deviations[j] = get_coord_of_min_dist(centroids[j], final_axis)


# define r'(t)
final_axis_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001)
# define r"(t) for later
final_axis_second_deriv = deriv_axis_spline_curve(new_avgs[:,0], new_avgs[:,1], new_avgs[:,2], 0.001, 2)
# define T(t) = r'(t)/||r'(t)||
r_prime_norms = np.repeat(np.expand_dims(np.linalg.norm(final_axis_deriv, axis=-1), axis=-1), 3, axis=-1)
T_of_t = final_axis_deriv / r_prime_norms

'''N_of_t = deriv_axis_spline_curve(T_of_t[:,0], T_of_t[:,1], T_of_t[:,2], 1)
N_of_t = N_of_t / np.linalg.norm(N_of_t, axis=0)

B_of_t = []
for i in range(0, len(T_of_t)):
	B_of_t.append(np.cross(T_of_t[i], N_of_t[i]))

B_of_t = np.asarray(B_of_t)
B_of_t = B_of_t / np.linalg.norm(B_of_t, axis=0)
'''

rots = []
for i in range(0, len(T_of_t)):
	rots.append(get_R_from_a_to_b(np.array([0,0,-1]), T_of_t[i]))
	#rots.append(np.linalg.solve([[0,0,1],[0,1,0],[1,0,0]], [T_of_t[i], N_of_t[i], B_of_t[i]]).T)

rots = np.asarray(rots)






