## To take into consideration the following issue with the code in main.py
### The clustering has a lot of noise. 
### Using variance as a cost function for new clusters means that we are assuming that the new clusters appeared out of nowhere, 
### that crimes started occurring in that cluster only in that year. 
### But that is not true, as it is just that in the crimes in the same locations in the previous year were relatively sparse.

data_2008 = None
data_2009 = None

from sklearn.preprocessing import StandardScaler

# Assuming data_2008 and data_2009 are pandas DataFrames with the same structure
## Preprocess Data
scaler = StandardScaler()
data_2008_normalized = scaler.fit_transform(data_2008)
data_2009_normalized = scaler.transform(data_2009)

## Cluster Data
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

def compute_centroid(cluster):
    return np.mean(cluster, axis=0)

def compute_size(cluster):
    return len(cluster)

def compute_variance(cluster):
    return np.var(cluster, axis=0).mean()

def compute_density(cluster, all_data, bandwidth=1.0):
    # Use a kernel density estimation or nearest neighbor approach to estimate density
    tree = cKDTree(all_data)
    densities = tree.query_ball_point(cluster, r=bandwidth) # r needs to be changed to an appropriate distance metric. 
    density = np.mean([len(d) for d in densities]) / (bandwidth ** len(cluster[0]))
    return density

def compute_cost_matrix(clusters_2008, clusters_2009, all_data_2008, all_data_2009, alpha=1, beta=1, gamma=1):
    num_clusters_2008 = len(clusters_2008)
    num_clusters_2009 = len(clusters_2009)
    
    # Initialize cost matrix with large values for dummy matches
    cost_matrix = np.full((num_clusters_2008 + num_clusters_2009, num_clusters_2008 + num_clusters_2009), np.inf)
    
    # Fill the actual cost matrix
    for i, c1 in enumerate(clusters_2008):
        for j, c2 in enumerate(clusters_2009):
            centroid_distance = np.linalg.norm(compute_centroid(c1) - compute_centroid(c2))
            size_difference = abs(compute_size(c1) - compute_size(c2))
            cost_matrix[i, j] = alpha * centroid_distance + beta * size_difference
    
    # Fill dummy costs for disappearing clusters based on their variance
    for i in range(num_clusters_2008):
        cost_matrix[i, num_clusters_2009 + i] = compute_variance(clusters_2008[i])
    
    # Fill dummy costs for new clusters based on density in the previous year's data
    for j in range(num_clusters_2009):
        density_cost = compute_density(clusters_2009[j], all_data_2008)
        cost_matrix[num_clusters_2008 + j, j] = gamma * density_cost
    
    return cost_matrix

def match_clusters(clusters_2008, clusters_2009, all_data_2008, all_data_2009):
    cost_matrix = compute_cost_matrix(clusters_2008, clusters_2009, all_data_2008, all_data_2009)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

# Example clusters as lists of numpy arrays
clusters_2008 = [np.random.rand(100, 2) for _ in range(5)]
clusters_2009 = [np.random.rand(100, 2) for _ in range(6)]
all_data_2008 = np.vstack(clusters_2008)
all_data_2009 = np.vstack(clusters_2009)

row_ind, col_ind = match_clusters(clusters_2008, clusters_2009, all_data_2008, all_data_2009)

## Calculate EMD
from pyemd import emd

def calculate_partial_emd(cluster1, cluster2):
    # Combine clusters into a single dataset for partial matching
    combined_cluster = np.vstack((cluster1, cluster2))
    num_points_1 = cluster1.shape[0]
    num_points_2 = cluster2.shape[0]
    
    # Create signatures with weights
    signature1 = np.hstack((cluster1, np.ones((num_points_1, 1)) / num_points_1))
    signature2 = np.hstack((cluster2, np.ones((num_points_2, 1)) / num_points_2))
    
    # Distance matrix
    distance_matrix = np.linalg.norm(signature1[:, None, :-1] - signature2[None, :, :-1], axis=2)
    
    # Calculate EMD with partial matching
    emd_distance = emd(signature1[:, -1], signature2[:, -1], distance_matrix)
    return emd_distance

emd_distances = []
for i, j in zip(row_ind, col_ind):
    if i < len(clusters_2008) and j < len(clusters_2009):
        emd_distance = calculate_partial_emd(clusters_2008[i], clusters_2009[j])
        emd_distances.append(emd_distance)
    elif i < len(clusters_2008):  # Disappearing cluster
        emd_distances.append(compute_variance(clusters_2008[i]))
    elif j < len(clusters_2009):  # New cluster
        density_cost = compute_density(clusters_2009[j], all_data_2008)
        emd_distances.append(density_cost)

print("EMD Distances between matched clusters:", emd_distances)
