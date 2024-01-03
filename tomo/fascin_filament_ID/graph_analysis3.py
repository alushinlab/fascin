#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
import networkx as nx
from networkx.algorithms.community import girvan_newman
from networkx.algorithms import isomorphism
import pickle
from scipy.stats import sem
from collections import Counter
import itertools
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from functools import partial
print('Imports finished. Beginning script...')
################################################################################
def sem(arr):
    return np.std(arr) / np.sqrt(len(arr))

def generate_triangular_tessellation(rows, cols, edge_weight=1):
    rows, cols = int(rows), int(cols)
    G = nx.Graph()
    for i in range(rows):
        for j in range(cols):
            # Add node to the graph
            node_id = i * cols + j
            G.add_node(node_id, pos=(j, -i))
            # Connect with node on the right
            if j < cols - 1:
                G.add_edge(node_id, node_id + 1, weight=edge_weight)
            # Connect with node below
            if i < rows - 1:
                G.add_edge(node_id, node_id + cols, weight=edge_weight)
            # Connect diagonally (bottom-right) to form triangles
            if i < rows - 1 and j < cols - 1:
                G.add_edge(node_id, node_id + cols + 1, weight=edge_weight)
    return G

def revertEdgeScores(G):
    G2 = G.copy()
    # Update edge weights
    for u, v in G2.edges():
        edge_weight_transformed = G2[u][v]['weight']
        if edge_weight_transformed <= 0:
            print(f"Warning: Non-positive edge weight {edge_weight_transformed} encountered between {u} and {v}. Setting energy_score_original to a large value.")
            energy_score_original = float(10.0)
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

def negateEdges(G):
    G2 = G.copy()
    # Iterate through edges in G to populate G2
    for u, v in G2.edges():
        G2[u][v]['weight'] = -1.0*G2[u][v]['weight']
    return G2
def compute_betweenness_scatters(G,G2, saving_path):
    betweenness_dict = nx.betweenness_centrality(G)
    betweenness_dict_weighted = nx.betweenness_centrality(G, weight='weight')
    weighted_degree_dict = dict(G.degree(weight='weight'))
    unweighted_degree_dict = dict(G.degree())
    average_weight_dict = {node: (weighted_degree_dict[node] / unweighted_degree_dict[node]) if unweighted_degree_dict[node] > 0 else 0 for node in G.nodes()}
    # Compute untransformed edge weights
    betweenness_dict_G2 = nx.betweenness_centrality(G2)
    betweenness_dict_weighted_G2 = nx.betweenness_centrality(G2, weight='weight')
    weighted_degree_dict_G2 = dict(G2.degree(weight='weight'))
    unweighted_degree_dict_G2 = dict(G2.degree())
    average_weight_dict_G2 = {node: (weighted_degree_dict_G2[node] / unweighted_degree_dict_G2[node]) if unweighted_degree_dict_G2[node] > 0 else 0 for node in G2.nodes()}
    
    # Prepare for scatter plot
    betweenness_list = list(betweenness_dict.values())
    weighted_betweenness_list = list(betweenness_dict_weighted.values())
    degree_list = list(unweighted_degree_dict.values())
    weighted_degree_list = list(weighted_degree_dict.values())
    average_weight_list = list(average_weight_dict.values())
    degree_list_G2 = list(unweighted_degree_dict_G2.values())
    weighted_degree_list_G2 = list(weighted_degree_dict_G2.values())
    average_weight_list_G2 = list(average_weight_dict_G2.values())
    betweenness_list_G2 = list(betweenness_dict_G2.values())
    weighted_betweenness_list_G2 = list(betweenness_dict_weighted_G2.values())

    # Scatter plot
    fig,ax = plt.subplots(2,2, figsize=(8,8))
    ax[0,0].scatter(betweenness_list_G2, weighted_degree_list_G2)
    ax[0,0].set_xlabel('Betweenness Centrality')
    ax[0,0].set_ylabel('Weighted degree')
    ax[0,1].scatter(betweenness_list_G2, average_weight_list_G2)
    ax[0,1].set_xlabel('Betweenness Centrality')
    ax[0,1].set_ylabel('Average weighted degree')
    ax[1,0].scatter(weighted_betweenness_list_G2, weighted_degree_list_G2)
    ax[1,0].set_xlabel('Weighted Betweenness Centrality')
    ax[1,0].set_ylabel('Weighted degree')
    ax[1,1].scatter(weighted_betweenness_list_G2, average_weight_list_G2)
    ax[1,1].set_xlabel('Weighted Betweenness Centrality')
    ax[1,1].set_ylabel('Average weighted degree')
    plt.tight_layout()
    plt.savefig(saving_path+'betweenness_centrality.png', format='png', dpi=300)
    plt.clf()
    fig,ax = plt.subplots(2,2, figsize=(8,8))
    ax[0,0].scatter(degree_list, average_weight_list)
    ax[0,0].set_xlabel('Degree')
    ax[0,0].set_ylabel('Average weighted degree')
    ax[0,1].scatter(degree_list, weighted_degree_list)
    ax[0,1].set_xlabel('Degree')
    ax[0,1].set_ylabel('Weighted degree')
    ax[1,0].scatter(degree_list_G2, average_weight_list_G2)
    ax[1,0].set_xlabel('Degree')
    ax[1,0].set_ylabel('Average weighted degree (untransformed)')
    ax[1,1].scatter(degree_list_G2, weighted_degree_list_G2)
    ax[1,1].set_xlabel('Degree')
    ax[1,1].set_ylabel('Weighted degree (untransformed)')
    plt.tight_layout()
    plt.savefig(saving_path+'degree_comparisons.png', format='png', dpi=300)
    plt.clf(); plt.close()

def edge_density(G):
    """
    Calculate edge density of the graph G.
    Formula: Edge Density = 2 * |E| / (|V| * (|V| - 1))
    """
    E = G.number_of_edges()
    V = G.number_of_nodes()
    if V * (V - 1) == 0:  # to avoid division by zero
        return 0
    return 2 * E / (V * (V - 1))

def process_fraction(frac, G, do_random_removal, n_runs, do_nodes):
    myG = G.copy()
    metrics_run = {
        "avg_size": np.zeros(n_runs),
        "num_components": np.zeros(n_runs),
        "transitivity": np.zeros(n_runs),
        "local_efficiency": np.zeros(n_runs),
        "clustering_coeff": np.zeros(n_runs),
        "edge_density": np.zeros(n_runs)
    }
    for i in range(n_runs):
        G_temp = myG.copy()
        if do_nodes:
            sizes, num_components = random_removal(G_temp, frac) if do_random_removal else targeted_removal(G_temp, frac)
        else:
            sizes, num_components = random_edge_removal(G_temp, frac) if do_random_removal else targeted_edge_removal(G_temp, frac)
        metrics_run["avg_size"][i] = np.mean(sizes)
        metrics_run["num_components"][i] = num_components
        metrics_run["transitivity"][i] = nx.transitivity(G_temp)
        metrics_run["local_efficiency"][i] = nx.local_efficiency(G_temp)
        metrics_run["clustering_coeff"][i] = nx.average_clustering(G_temp)
        metrics_run["edge_density"][i] = edge_density(G_temp)
    return {k: np.mean(v) for k, v in metrics_run.items()}, {k+"_sem": sem(v) * 1.96 for k, v in metrics_run.items()}

def probe_resiliency(graphs, labels, saving_path, do_nodes):
    fractions = np.linspace(0, 0.9, 45)
    n_runs = 50
    results = {}
    metrics_list = ["avg_size", "num_components", "transitivity", "local_efficiency", "clustering_coeff", "edge_density"]
    for label, G in zip(["Random Removal"] + labels, [graphs[0]] + graphs):
        with ThreadPoolExecutor() as executor:
           run_results = list(executor.map(lambda frac: process_fraction(frac, G, label == "Random Removal", n_runs, do_nodes), fractions))
        results[label] = {"means": [res[0] for res in run_results], "sems": [res[1] for res in run_results]}

    # Conversion & Plotting logic
    fig, ax = plt.subplots(len(metrics_list)//2, 2, figsize=(12, 12))
    ax = ax.flatten()
    for i, metric in enumerate(metrics_list):
        for label, res in results.items():
            mean_values = np.array([d[metric] for d in res["means"]])
            sem_values = np.array([d[metric+"_sem"] for d in res["sems"]])
            ax[i].errorbar(fractions, mean_values, yerr=sem_values, label=f'{label}', alpha=0.7)
            ax[i].fill_between(fractions, mean_values - sem_values, mean_values + sem_values, alpha=0.2)
            ax[i].set_xlabel('Fraction removed')
            ax[i].set_ylabel(metric.replace('_', ' ').capitalize())
    for sub_ax in ax:
        sub_ax.legend()
    plt.tight_layout()
    filename = 'probeResiliency.png' if do_nodes else 'probeResiliency_edges.png'
    plt.savefig(saving_path+filename, format='png', dpi=300)
    plt.clf(); plt.close()

def random_removal(G, frac):
    nodes = list(G.nodes())
    num_remove = int(frac * len(nodes))
    remove_nodes = np.random.choice(nodes, num_remove, replace=False)
    G.remove_nodes_from(remove_nodes)
    components = [len(c) for c in nx.connected_components(G)]
    return components, len(components)

def targeted_removal2(G, frac):
    nodes = list(G.nodes())
    num_remove = int(frac * len(nodes))
    probabilities = []
    for n in nodes:
        s = sum(weight for _, _, weight in G.edges(n, data='weight'))
        if np.isnan(s):
            probabilities.append(0)  # set probability to zero for these nodes
        else:
            probabilities.append(s)
    probabilities = np.array(probabilities) / sum(probabilities)
    remove_nodes = np.random.choice(nodes, num_remove, replace=False, p=probabilities)
    G.remove_nodes_from(remove_nodes)
    components = [len(c) for c in nx.connected_components(G)]
    return components, len(components)

def targeted_removal(G, frac):
    nodes = list(G.nodes())
    num_remove = int(frac * len(nodes))
    sums = []
    for n in nodes:
        s = sum(weight for _, _, weight in G.edges(n, data='weight'))
        if np.isnan(s):
            sums.append(0)  # set sum to zero for these nodes
        else:
            sums.append(s)
    # Get the ranks of the sums
    ranks = np.argsort(np.argsort(sums))
    # Use the ranks as probabilities
    probabilities = ranks / sum(ranks)
    # If all ranks are zero (i.e., all sums are zero or nan), then probabilities will become nan. In that case, replace with uniform probabilities.
    if np.any(np.isnan(probabilities)):
        probabilities = np.ones_like(probabilities) / len(probabilities)
    remove_nodes = np.random.choice(nodes, num_remove, replace=False, p=probabilities)
    G.remove_nodes_from(remove_nodes)
    components = [len(c) for c in nx.connected_components(G)]
    return components, len(components)

def random_edge_removal(G, frac):
    edges = list(G.edges())
    num_remove = int(frac * len(edges))
    edge_to_index = {edge: index for index, edge in enumerate(edges)}
    edge_indices = np.array(list(edge_to_index.values()))
    chosen_indices = np.random.choice(edge_indices, num_remove, replace=False)
    chosen_edges = [edges[i] for i in chosen_indices]
    G.remove_edges_from(chosen_edges)
    components = [len(c) for c in nx.connected_components(G)]
    return components, len(components)

def targeted_edge_removal2(G, frac):
    edges = list(G.edges())
    num_remove = int(frac * len(edges))
    edge_to_index = {edge: index for index, edge in enumerate(edges)}
    edge_indices = np.array(list(edge_to_index.values()))
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    probabilities = np.array(edge_weights) / sum(edge_weights)
    chosen_indices = np.random.choice(edge_indices, num_remove, replace=False, p=probabilities)
    chosen_edges = [edges[i] for i in chosen_indices]
    G.remove_edges_from(chosen_edges)
    components = [len(c) for c in nx.connected_components(G)]
    return components, len(components)

def targeted_edge_removal(G, frac):
    edges = list(G.edges())
    num_remove = int(frac * len(edges))
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    # Get the ranks of the edge_weights
    ranks = np.argsort(np.argsort(edge_weights))
    # Use the ranks as probabilities
    probabilities = ranks / sum(ranks)
    # If all ranks are zero (i.e., all edge_weights are zero or nan), then probabilities will become nan. In that case, replace with uniform probabilities.
    if np.any(np.isnan(probabilities)):
        probabilities = np.ones_like(probabilities) / len(probabilities)
    chosen_indices = np.random.choice(range(len(edges)), num_remove, replace=False, p=probabilities)
    chosen_edges = [edges[i] for i in chosen_indices]
    G.remove_edges_from(chosen_edges)
    components = [len(c) for c in nx.connected_components(G)]
    return components, len(components)

def is_hexagon(neighbors, G):
    n = len(neighbors)
    for i in range(n):
        if not G.has_edge(neighbors[i], neighbors[(i + 1) % n]):
            return False
    return True

def find_hexagons(G):
    hexagon_nodes = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 6:
            continue
        # itertools.combinations will give us unique combinations of 6 neighbors
        for combo in itertools.permutations(neighbors, 6):
            if is_hexagon(combo, G):
                hexagon_nodes.append(node)
                break  # No need to check further combinations for this node
    return hexagon_nodes

def find_triangles(G):
    triangles = set()
    visited_nodes = set()  # To store nodes already part of a triangle
    for node in G.nodes():
        if node in visited_nodes:
            continue
        neighbors = set(G.neighbors(node))
        if len(neighbors) < 2:
            continue  # Skip nodes with fewer than 2 neighbors
        for neighbor1 in neighbors:
            for neighbor2 in neighbors:
                if neighbor1 >= neighbor2:
                    continue  # Skip duplicate combinations
                if G.has_edge(neighbor1, neighbor2):
                    # Sort the nodes to have a unique identifier for each triangle
                    triangle_nodes = tuple(sorted([node, neighbor1, neighbor2]))
                    if triangle_nodes not in triangles:
                        triangles.add(triangle_nodes)
                        visited_nodes.update(triangle_nodes)
    return triangles

def find_diamonds(G):
    triangles = find_triangles(G)
    diamonds = set()
    for t1 in triangles:
        for t2 in triangles:
            # Skip the same triangle
            if t1 == t2:
                continue
            # Find the common edges
            common_nodes = set(t1).intersection(set(t2))
            if len(common_nodes) == 2:
                # Sort and add to diamonds set
                diamond = tuple(sorted(set(t1).union(set(t2))))
                if len(diamond) == 4:
                    diamonds.add(diamond)
    return diamonds

def remove_low_weight_edges(G, threshold):
    G2 = G.copy()
    edges_to_remove = []
    for u, v, data in G2.edges(data=True):
        if data['weight'] < threshold:
            edges_to_remove.append((u, v))
    G2.remove_edges_from(edges_to_remove)
    return G2

################################################################################
graph_base_name = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/automated_fil_identification_curated/'
pkl_folder_names = ['ts044', 'ts045', 'ts053', 'ts058', 'ts059', 'ts062', 'ts064', 'ts065', 'ts072', 'ts074']
compiled_stats_holder = []
edge_weights_holder = []
for i in tqdm(range(0, len(pkl_folder_names))):
    file_name = graph_base_name + pkl_folder_names[i] + '/graph_theory/graph.pkl'
    saving_path = os.path.dirname(file_name) + '/'
    with open(file_name, 'rb') as f:
        G = pickle.load(f)
    
    G2 = revertEdgeScores(G) # Reverts back to the filament interface scores, higher score is stronger connection
    G3 = setEdgeScoresToOne(G) # Set uniform edge weights
    G4 = negateEdges(G2) # Negate Filament interface scores from original scores
    G5 = negateEdges(G3) # Negate uniform edge weights
    #G4 = remove_low_weight_edges(G2, 1.0)
    #compute_betweenness_scatters(G,G2, saving_path)
    graph_list = [G2, G4, G3, G5]
    labels = ['Remove Best First (RBF)', 'Remove Worst First (RWF)', 'Uniform Edge Weight RBF', 'Uniform Edge Weight RWF']
    print("Number of connected components: " + str(len(list(nx.connected_components(G)))) + ' ' + str(len(list(nx.connected_components(G2)))) + ' ' + str(len(list(nx.connected_components(G3)))) + ' ')
    probe_resiliency(graph_list, labels, saving_path, True)
    probe_resiliency(graph_list, labels, saving_path, False)
    #triangular_tess = generate_triangular_tessellation(np.sqrt(len(G.nodes())), np.sqrt(len(G.nodes())), edge_weight=1)
    #probe_resiliency(triangular_tess, saving_path+'perfectTiling.png')
    full_bundles = nx.connected_components(G2)
    graph_stats = []
    for component in full_bundles:
        subgraph = G2.subgraph(component)
        edge_weights = [data['weight'] for u, v, data in subgraph.edges(data=True)]
        edge_weights_holder.extend(edge_weights)
        num_nodes = len(subgraph.nodes())
        num_edges = len(subgraph.edges())
        total_weight = sum(data['weight'] for u, v, data in subgraph.edges(data=True))
        avg_weight = total_weight / num_edges if num_edges > 0 else 0
        # Store these metrics
        graph_stats = ((num_nodes, num_edges, avg_weight, total_weight))
        compiled_stats_holder.append(graph_stats)

compiled_stats_holder = np.asarray(compiled_stats_holder)
fig, axs = plt.subplots(2, 2, figsize=(10,10))
# Plot Number of Nodes vs Number of Edges
axs[0,0].scatter(compiled_stats_holder[:,0], compiled_stats_holder[:,1])
axs[0,0].set_xlabel('Number of Nodes')
axs[0,0].set_ylabel('Number of Edges')
axs[0,0].set_title('Nodes vs Edges')
# Plot Number of Nodes vs Average Weight
axs[0,1].scatter(compiled_stats_holder[:,0], compiled_stats_holder[:,2])
axs[0,1].set_xlabel('Number of Nodes')
axs[0,1].set_ylabel('Average Weight')
axs[0,1].set_title('Nodes vs Avg Weight')
# Plot Number of Nodes vs Total Weighted Degree
axs[1,0].scatter(compiled_stats_holder[:,0], compiled_stats_holder[:,3])
axs[1,0].set_xlabel('Number of Nodes')
axs[1,0].set_ylabel('Total Weighted Degree')
axs[1,0].set_title('Nodes vs Total Weighted Degree')
# Generate histogram for edge weights
axs[1,1].hist(edge_weights_holder, bins=100)
axs[1,1].set_xlabel('Edge Weight')
axs[1,1].set_xlim(0,7)
axs[1,1].set_ylabel('Frequency')
axs[1,1].set_title('Histogram of Edge Weights Across All Graphs')
# Save figure
plt.tight_layout()
plt.savefig(graph_base_name+'compiled_metrics.png', format='png', dpi=300)

print('Hi')






