#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import networkx as nx
import pickle
import matplotlib.animation as animation
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from matplotlib.animation import FFMpegWriter
from networkx.algorithms import community
print('Imports finished. Beginning script...')
################################################################################
def generate_triangular_tessellation(rows, cols, edge_weight=1):
    rows, cols = int(rows), int(cols)
    weights = np.random.normal(3, 0.2,(rows,cols))
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            # Add node to the graph
            node_id = i * cols + j
            G.add_node(node_id, pos=(j, -i))
            # Connect with node on the right
            if j < cols - 1:
                G.add_edge(node_id, node_id + 1, weight=weights[i][j])
            # Connect with node below
            if i < rows - 1:
                G.add_edge(node_id, node_id + cols, weight=weights[i][j])
            # Connect diagonally (bottom-right) to form triangles
            if i < rows - 1 and j < cols - 1:
                G.add_edge(node_id, node_id + cols + 1, weight=weights[i][j])
    return G

def save_cluster_img(G, saving_path):
    components = list(nx.connected_components(G))
    for idx, nodes in enumerate(components):
        subgraph = G.subgraph(nodes)
        pos = nx.spring_layout(subgraph)  # Compute node positions
        colors = [subgraph.nodes[node]['cluster'] for node in subgraph.nodes()]
        plt.figure(figsize=(10, 10))
        nx.draw(subgraph, pos, with_labels=True, node_color=colors, cmap=plt.cm.jet)
        plt.title(f'Cluster, Component {idx + 1}')
        # Save the image
        image_path = os.path.join(saving_path, f'cluster_plot__component_{idx + 1}_notMulti.png')
        plt.savefig(image_path)
        plt.close()

def optimal_num_clusters(G, saving_path, max_clusters=None):
    A = nx.adjacency_matrix(G, weight='weight')
    degrees = np.array(G.degree(weight='weight'))
    D = np.diag(degrees[:, 1])
    L = D - A.toarray()

    if max_clusters is None:
        max_clusters = len(G.nodes())
        
    eigenvalues = np.sort(np.real(np.linalg.eigvals(L)))[:max_clusters]
    eigengaps = np.diff(eigenvalues)
    fig, ax = plt.subplots(2)
    ax[0].plot(eigenvalues)
    ax[1].plot(eigengaps)
    ax[1].scatter(np.argmax(eigengaps), eigengaps[np.argmax(eigengaps)])
    plt.savefig(saving_path+'spectral_clustering_evals.png')
    plt.close()
    return np.argmax(eigengaps) + 1

def get_global_max_similarity(Gs):
    similarity_holder = []
    for k in range(0, len(Gs)):
        # Create a similarity matrix
        num_nodes = len(Gs[k].nodes())
        similarity_matrix = np.zeros((num_nodes, num_nodes))
        
        for i, node_i in enumerate(Gs[k].nodes()):
            for j, node_j in enumerate(Gs[k].nodes()):
                if i != j and Gs[k].has_edge(node_i, node_j):
                    # Set similarity as edge weight
                    similarity_matrix[i][j] = Gs[k][node_i][node_j]['weight']
        this_max_similarity = np.max(similarity_matrix[similarity_matrix != 0])
        similarity_holder.append(this_max_similarity)
    return np.max(similarity_holder)

def conductance(G, S):
    cut = nx.cut_size(G, S)
    volume = sum(d for n, d in G.degree(S))
    return cut / volume

def hierarchical_clustering_and_dendrogram(G, saving_path, max_similarity):
    # Create a similarity matrix
    num_nodes = len(G.nodes())
    similarity_matrix = np.zeros((num_nodes, num_nodes))
    
    for i, node_i in enumerate(G.nodes()):
        for j, node_j in enumerate(G.nodes()):
            if i != j and G.has_edge(node_i, node_j):
                # Set similarity as edge weight
                similarity_matrix[i][j] = G[node_i][node_j]['weight']

    max_similarity = max_similarity#np.max(similarity_matrix[similarity_matrix != 0])  # Avoid dividing by zero
    distance_matrix = 1 - (similarity_matrix / max_similarity)
    np.fill_diagonal(distance_matrix, 0)    # Convert the similarity matrix to a condensed distance matrix
    condensed_distance_matrix = sch.distance.squareform(distance_matrix, checks=True)

    # Generate the linkage matrix
    linkage_matrix = sch.linkage(condensed_distance_matrix, method='ward')  # Don't use complete or single linkage
    max_linkage_distance = linkage_matrix[-1, 2]+0.05 # add extra to get full connections all the way
    distance_cutoffs = np.arange(0.0, max_linkage_distance, 0.01)  # adjust the step size as necessary 0.1 is good
    all_transitivities, all_densities, all_sizes = [], [], []
    all_num_connected_components, all_avg_clustering_coeff = [], []
    all_largest_components, all_smallest_components = [], []
    all_modularity = []
    all_avg_diameters  = []
    for cutoff in distance_cutoffs:
        # Use the fcluster function to obtain clusters at the given cutoff
        cluster_labels = sch.fcluster(linkage_matrix, cutoff, criterion='distance')
        
        # Convert cluster labels to clusters of nodes
        nodes = list(G.nodes())
        clusters = {i: [] for i in set(cluster_labels)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(nodes[idx])

        # Compute properties for each cluster and average them
        transitivities, densities, sizes, clustering_coefficients, diameters = [], [], [], [], []
        for cluster in clusters.values():
            subgraph = G.subgraph(cluster)
            transitivities.append(nx.transitivity(subgraph))
            densities.append(nx.density(subgraph))
            sizes.append(subgraph.number_of_nodes())
            clustering_coefficients.append(nx.average_clustering(subgraph, weight='weight'))
            if nx.is_connected(subgraph):  # Make sure the subgraph is connected
                bc_values = nx.betweenness_centrality(subgraph).values()
                diameters.append(np.mean(list(bc_values)))
        
        all_transitivities.append(np.mean(transitivities))
        all_densities.append(np.mean(densities))
        all_sizes.append(np.mean(sizes))
        all_num_connected_components.append(len(clusters))
        all_avg_clustering_coeff.append(np.mean(clustering_coefficients))
        all_largest_components.append(max(sizes))
        all_smallest_components.append(min(sizes))
        all_avg_diameters.append(np.mean(diameters) if diameters else 0)
        communities = list(clusters.values())
        modularity_value = community.modularity(G,communities)
        all_modularity.append(modularity_value)

    merge_distances = linkage_matrix[:,2]
    sorted_merge_distances = np.sort(merge_distances)
    gaps = np.diff(sorted_merge_distances)

    all_avg_clustering_coeff = all_avg_clustering_coeff / np.max(all_avg_clustering_coeff)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    sch.dendrogram(linkage_matrix, ax=axs[0, 0])
    axs[0, 0].set_title('Dendrogram')
    axs[0, 1].plot(distance_cutoffs, all_transitivities, marker='o')
    axs[0, 1].set_title('Transitivity vs Distance Cutoff')
    axs[0, 2].plot(distance_cutoffs, all_densities, marker='o')
    axs[0, 2].set_title('Density vs Distance Cutoff')
    axs[0, 3].plot(distance_cutoffs, all_sizes, marker='o')
    axs[0, 3].plot(distance_cutoffs, all_largest_components, marker='o')
    axs[0, 3].plot(distance_cutoffs, all_smallest_components, marker='o')
    axs[0, 3].set_title('Average Bundle Size vs Distance Cutoff')
    axs[1, 0].plot(distance_cutoffs, all_num_connected_components, marker='o')
    axs[1, 0].set_title('Number of Connected Components vs Distance Cutoff')
    axs[1, 1].plot(distance_cutoffs, all_avg_clustering_coeff, marker='o')
    axs[1, 1].set_title('Avg Clustering Coefficient vs Distance Cutoff')
    axs[1, 2].plot(distance_cutoffs, all_avg_diameters, marker='o')
    axs[1, 2].set_title('Average Betweenness Centrality vs Distance Cutoff')
    axs[1, 3].plot(distance_cutoffs, all_modularity, marker='o')
    axs[1, 3].set_title('Modularity vs Distance Cutoff')
    axs[0,0].set_ylim([0.0,1.8])
    #axs[0,0].set_yscale('log')
    axs[0,1].set_ylim([0.0,1.0])
    axs[0,2].set_ylim([0.0,1.0])
    axs[1,1].set_ylim([0.0,1.0])
    axs[0,3].set_yscale('log')
    axs[0,3].set_ylim(1, 250)
    axs[1,3].set_ylim(0, 1.0)
    axs[1,2].set_ylim(0, 0.15)
    for ax in axs.flat[1:]:
        ax.set_xlim([0.0, 1.8]) #1.6 for weighted; 2.5 for unweighted
    plt.tight_layout()
    plt.savefig(saving_path+ 'hierarchical_dendrogram.png')
    plt.savefig(saving_path+ 'hierarchical_dendrogram.svg')
    plt.close()
    #generate_animation(G, linkage_matrix, distance_cutoffs, saving_path)
    return G, linkage_matrix, [distance_cutoffs, all_densities, all_transitivities, all_avg_clustering_coeff, all_avg_diameters,all_modularity,all_sizes, all_largest_components, all_smallest_components]

def revertEdgeScores(G):
    G2 = G.copy()
    # Update edge weights
    for u, v in G2.edges():
        edge_weight_transformed = G2[u][v]['weight']
        if edge_weight_transformed <= 0:
            print(f"Warning: Non-positive edge weight {edge_weight_transformed} encountered between {u} and {v}. Setting energy_score_original to a large value.")
            energy_score_original = float(10.0)
            print('Problem areas:')
            print(G2.nodes[u]['filament_path'])
            print(G2.nodes[v]['filament_path'])
        else:
            energy_score_original = -np.log(edge_weight_transformed)
        G2[u][v]['weight'] = energy_score_original
    return G2

def setEdgeScoresToOne(G):
    G2 = G.copy()
    # Iterate through edges in G to populate G2
    for u, v in G2.edges():
        G2[u][v]['weight'] = 1
    return G2

def scramble_edge_weights(G):
    G2 = G.copy()
    weights = [data['weight'] for _, _, data in G2.edges(data=True)]
    np.random.shuffle(weights)
    G_scrambled = G2.copy()
    for idx, (_, _, data) in enumerate(G_scrambled.edges(data=True)):
        data['weight'] = weights[idx]
    return G_scrambled

def extract_subgraphs_at_cutoff(G, linkage_matrix, cutoff):
    cluster_labels = sch.fcluster(linkage_matrix, t=cutoff, criterion='distance')
    nodes = list(G.nodes())
    clusters = {i: [] for i in set(cluster_labels)}
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(nodes[idx])
    
    subgraphs = [G.subgraph(cluster) for cluster in clusters.values()]
    return subgraphs

def generate_animation(G, linkage_matrix, distance_cutoffs, saving_path):
    fig, ax = plt.subplots()
    # Compute a global position layout once for the whole graph
    global_pos = nx.spring_layout(G)
    # List of colors to use for subgraphs
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # Add more if needed
    def animate(cutoff):
        ax.clear()
        subgraphs = extract_subgraphs_at_cutoff(G, linkage_matrix, cutoff)
        # Initialize an offset to help space out the graphs
        offset_x = 0
        for idx, subgraph in enumerate(subgraphs):
            # Filter the global positions for nodes in the subgraph
            subgraph_pos = {node: (x + offset_x, y) for node, (x, y) in global_pos.items() if node in subgraph}
            # Calculate the width of this subgraph based on the x coordinates and add some spacing
            width_of_current_subgraph = max(x for x, y in subgraph_pos.values()) - min(x for x, y in subgraph_pos.values())
            offset_x += width_of_current_subgraph + 1.0  # The value '1.0' is an arbitrary spacing value, you can adjust it
            # Draw using the color cycle
            nx.draw(subgraph, subgraph_pos, ax=ax, node_color=colors[idx % len(colors)])
    ani = animation.FuncAnimation(fig, animate, frames=distance_cutoffs, repeat=False)
    writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(os.path.join(saving_path, 'animation.mp4'), writer=writer)

def plot_graph(G, saving_path):
    # Draw the graph
    pos = nx.spring_layout(G)  # position for all nodes, you can try different layouts
    plt.figure(figsize=(12, 12))  # set the figure size
    
    nx.draw_networkx_nodes(G, pos, node_size=500)  # draw nodes
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)  # draw edges
    nx.draw_networkx_labels(G, pos, font_size=12)  # draw node labels

    plt.title("Graph Visualization")
    plt.axis("off")  # turn off axis
    plt.tight_layout()
    plt.savefig(saving_path+ '_graph.png')
    plt.close()

def generate_distinct_colors(n):
    """Generate a list of distinct RGB colors."""
    np.random.seed(46)  # Setting seed for reproducibility
    colors = np.random.rand(n, 3)
    return colors

import xml.etree.ElementTree as ET
def change_cmm_color(input_filename, output_filename, new_r, new_g, new_b):
    # Parse the XML from the input file
    tree = ET.parse(input_filename)
    root = tree.getroot()
    markers_to_remove = []
    # Iterate through all markers and update their RGB values
    for marker in root.findall('marker'):
        x, y, z = float(marker.get('x')), float(marker.get('y')), float(marker.get('z'))
        marker.set('r', str(new_r))
        marker.set('g', str(new_g))
        marker.set('b', str(new_b))
        marker.set('radius', str(7))

    for marker in markers_to_remove:
        root.remove(marker)
    # Write the updated XML data to the output file
    tree.write(output_filename, encoding='utf-8', xml_declaration=True)

def insert_after_last_slash(filepath, insertion):
    head, tail = filepath.rsplit('/', 1)
    return f"{head}/{insertion}{tail}"

def find_peaks_in_metrics(metrics):
    dist_cutoffs, densities, transitivities, avg_clustering_coeff, avg_diameters, modularity, _, _, _ = metrics
    # Find the index of the maximum value for each metric
    max_density_index = np.argmax(densities)
    max_trans_index = np.argmax(transitivities)
    max_cluster_index = np.argmax(avg_clustering_coeff)
    max_diameter_index = np.argmax(avg_diameters)
    max_modularity_index = np.argmax(modularity)
    # Find the distance_cutoff value corresponding to the maximum metric values
    density_cutoff = dist_cutoffs[max_density_index]
    trans_cutoff = dist_cutoffs[max_trans_index]
    cluster_cutoff = dist_cutoffs[max_cluster_index]
    diameter_cutoff = dist_cutoffs[max_diameter_index]
    modularity_cutoff = dist_cutoffs[max_modularity_index]
    return [density_cutoff, trans_cutoff, cluster_cutoff, diameter_cutoff, modularity_cutoff], ['density_'+str(density_cutoff)[:5], 'transitivity_'+str(trans_cutoff)[:5], 'cluster_coeff_'+str(cluster_cutoff)[:5], 'betweeness_'+str(diameter_cutoff)[:5], 'modularity_'+str(modularity_cutoff)[:5]]


################################################################################
graph_base_name = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/automated_fil_identification_stable_cores_forFig/'
#pkl_folder_names = ['ts044', 'ts045', 'ts053', 'ts058', 'ts059', 'ts062', 'ts064', 'ts065', 'ts072', 'ts074']
#pkl_folder_names = ['ts074']#, 'ts045', 'ts053', 'ts058', 'ts059', 'ts062', 'ts064', 'ts065', 'ts072', 'ts074']
pkl_folder_names = ['ts044_A', 'ts045_A', 'ts045_B', 'ts045_C', 'ts045_D', 'ts053_A', 'ts053_B', 'ts058_A', 'ts058_B','ts058_C','ts059_A','ts059_B','ts059_C','ts059_D','ts062_A','ts064_A','ts064_B','ts064_C','ts065_A','ts072_A','ts072_C','ts074_A','ts074_B','ts074_C']
#pkl_folder_names = ['ts044_A']#, 'ts045_A', 'ts045_B', 'ts045_C', 'ts045_D', 'ts053_A', 'ts053_B', 'ts058_A', 'ts058_B','ts058_C','ts059_A','ts059_B','ts059_C','ts059_D','ts062_A','ts064_A','ts064_B','ts064_C','ts065_A','ts072_A','ts072_C','ts074_A','ts074_B','ts074_C']
all_graphs = []
for i in tqdm(range(0, len(pkl_folder_names))):
    file_name = graph_base_name + pkl_folder_names[i] + '/graph_theory/graph.pkl'
    saving_path = os.path.dirname(file_name) + '/'
    with open(file_name, 'rb') as f:
        G = pickle.load(f)
    G_reverted = revertEdgeScores(G)
    G2 = setEdgeScoresToOne(G)
    connected_subgraphs = [G_reverted.subgraph(c).copy() for c in nx.connected_components(G_reverted)]
    all_graphs.append(connected_subgraphs) #G2 for unweighted; G_reverted for weighted

for i in range(0, len(all_graphs)):
    print(len(all_graphs[i][0].nodes()))
overall_metrics = []
max_similarity = get_global_max_similarity([item for sublist in all_graphs for item in sublist])
for i in tqdm(range(0, len(all_graphs))):
    for j in range(0, len(all_graphs[i])):
        if(len(all_graphs[i][j].nodes()) >= 5):
            file_name = graph_base_name + pkl_folder_names[i] + '/graph_theory/graph.pkl'
            saving_path = os.path.dirname(file_name) + '/data' + str(j)
            #all_graphs[i][j] = scramble_edge_weights(all_graphs[i][j])
            G, linkage_matrix, metrics = hierarchical_clustering_and_dendrogram(all_graphs[i][j], saving_path, max_similarity)
            cutoff_dists, metric_names = find_peaks_in_metrics(metrics)
            #cutoff_dists = [0.9, 1.0,1.1,1.2]
            for k in range(0, len(cutoff_dists)):
                new_cmm_folder_name = graph_base_name + pkl_folder_names[i] + '/clustered_nodes_%s/'%str(metric_names[k])
                os.makedirs(new_cmm_folder_name, exist_ok=True)
                subgraphs_cutoff_one = extract_subgraphs_at_cutoff(G,linkage_matrix,cutoff_dists[k])
                rgbs = generate_distinct_colors(len(subgraphs_cutoff_one))
                for l in range(0, len(subgraphs_cutoff_one)):
                    for node, data in subgraphs_cutoff_one[l].nodes(data=True):
                        change_cmm_color(data['filament_path'], insert_after_last_slash(data['filament_path'], 'clustered_nodes_%s/'%str(metric_names[k])), rgbs[l][0], rgbs[l][1], rgbs[l][2])
            
            overall_metrics.append(metrics)

#[distance_cutoffs, all_densities, all_transitivities, all_avg_clustering_coeff, all_avg_diameters,sorted_merge_distances[:-1], gaps]
def average_data_with_stats(data_sets):
    max_length = max(len(d) for d in data_sets)
    filled_data = []
    for data in data_sets:
        filled = list(data) + [np.nan] * (max_length - len(data))
        filled_data.append(filled)
    arr = np.array(filled_data)
    avg = np.nanmean(arr, axis=0)
    std_dev = np.nanstd(arr, axis=0)
    sem = std_dev / np.sqrt(np.count_nonzero(~np.isnan(arr), axis=0))
    return avg.tolist(), sem.tolist(), std_dev.tolist()

all_distance_cutoffs = []
all_densities = []
all_transitivities = []
all_avg_clustering_coeff = []
all_avg_diameters = []
all_modularities = []
all_sizes = []
all_sizes_largest = []
all_sizes_smallest = []
for i in range(0, len(overall_metrics)):
    distance_cutoffs, densities, transitivities, avg_clustering_coeff, avg_diameters, avg_modularities, avg_sizes, avg_sizes_large, avg_sizes_small = overall_metrics[i]
    all_distance_cutoffs.append(distance_cutoffs)
    all_densities.append(densities)
    all_transitivities.append(transitivities)
    all_avg_clustering_coeff.append(avg_clustering_coeff)
    all_avg_diameters.append(avg_diameters)
    all_modularities.append(avg_modularities)
    all_sizes.append(avg_sizes)
    all_sizes_largest.append(avg_sizes_large)
    all_sizes_smallest.append(avg_sizes_small)

avg_distance_cutoffs, _,_ = average_data_with_stats(all_distance_cutoffs)
avg_densities, density_SEM, density_STD = average_data_with_stats(all_densities)
avg_transitivities, trans_SEM, trans_STD = average_data_with_stats(all_transitivities)
avg_clustering_coeff, clust_SEM, clust_STD = average_data_with_stats(all_avg_clustering_coeff)
avg_diameters, diam_SEM, diam_STD = average_data_with_stats(all_avg_diameters)
avg_mod, mod_SEM, mod_STD = average_data_with_stats(all_modularities)
avg_size, size_SEM, size_STD = average_data_with_stats(all_sizes)
avg_size_large, size_SEM_large, size_STD_large = average_data_with_stats(all_sizes_largest)
avg_size_small, size_SEM_small, size_STD_small = average_data_with_stats(all_sizes_smallest)


fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].set_title('Average Bundle Size vs Distance Cutoff')
#axs[0, 0].fill_between(avg_distance_cutoffs, np.asarray(avg_size)-np.asarray(size_STD), np.asarray(avg_size)+np.asarray(size_STD), alpha=0.5)
axs[0, 0].plot(avg_distance_cutoffs, avg_size)
axs[0, 0].plot(avg_distance_cutoffs, avg_size_large)
axs[0, 0].errorbar(avg_distance_cutoffs, avg_size_small)
axs[0, 1].set_title('Density vs Distance Cutoff')
axs[0, 1].errorbar(avg_distance_cutoffs, avg_densities, yerr=density_STD, marker='o', capsize=1)
axs[1, 0].set_title('Transitivity vs Distance Cutoff') # can just as easily replace
axs[1, 0].fill_between(avg_distance_cutoffs, np.asarray(avg_transitivities)-np.asarray(trans_STD), np.asarray(avg_transitivities)+np.asarray(trans_STD), alpha=0.5)
axs[1, 0].plot(avg_distance_cutoffs, avg_transitivities)
axs[1, 1].set_title('Average Modularity')
axs[1, 1].fill_between(avg_distance_cutoffs, np.asarray(avg_mod)-np.asarray(mod_STD), np.asarray(avg_mod)+np.asarray(mod_STD), alpha=0.5)
axs[1, 1].plot(avg_distance_cutoffs, avg_mod)


axs[0,0].set_yscale('log')
axs[0,0].set_ylim(1, 250)
axs[0,1].set_ylim([0.0,1.0])
axs[1,0].set_ylim([0.0,1.0])
axs[1,1].set_ylim(0, 1.0)
for ax in axs.flat:
    ax.set_xlim([0.0, 1.8]) #1.6 for weighted; 2.5 for unweighted
plt.tight_layout()
plt.savefig(graph_base_name + 'average_graphs.png')
plt.savefig(graph_base_name + 'average_graphs.svg')
plt.close()

stacked_params = np.asarray([avg_distance_cutoffs,avg_size, avg_size_large, avg_size_small, avg_densities, np.asarray(density_STD), np.asarray(avg_transitivities),np.asarray(trans_STD),np.asarray(avg_mod),np.asarray(mod_STD)])
np.savetxt(graph_base_name + 'average_graphs.csv', stacked_params)

'''
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].set_title('Average Bundle Size vs Distance Cutoff')
axs[0, 0].errorbar(avg_distance_cutoffs, avg_size, yerr=size_STD, marker='o', capsize=1)
axs[0, 0].errorbar(avg_distance_cutoffs, avg_size_large, yerr=size_STD_large, marker='o', capsize=1)
axs[0, 0].errorbar(avg_distance_cutoffs, avg_size_small, yerr=size_STD_small, marker='o', capsize=1)
axs[0, 1].set_title('Density vs Distance Cutoff')
axs[0, 1].errorbar(avg_distance_cutoffs, avg_densities, yerr=density_STD, marker='o', capsize=1)
axs[1, 0].set_title('Transitivity vs Distance Cutoff') # can just as easily replace
axs[1, 0].errorbar(avg_distance_cutoffs, avg_transitivities, yerr=trans_STD, marker='o', capsize=1)
axs[1, 1].set_title('Average Modularity')
axs[1, 1].errorbar(avg_distance_cutoffs, avg_mod, yerr=mod_STD, marker='o', capsize=1)

axs[0,0].set_yscale('log')
axs[0,0].set_ylim(1, 250)
axs[0,1].set_ylim([0.0,1.0])
axs[1,0].set_ylim([0.0,1.0])
axs[1,1].set_ylim(0, 1.0)
for ax in axs.flat[1:]:
    ax.set_xlim([0.0, 1.8]) #1.6 for weighted; 2.5 for unweighted
plt.tight_layout()
plt.savefig(graph_base_name + 'average_graphs.png')
plt.savefig(graph_base_name + 'average_graphs.svg')
plt.close()
'''
print('Hi')






