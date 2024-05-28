import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import os
import sys
if __name__=='__main__':
    if __package__ is None:
        import os.path as path
        sys.path.insert(0, path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
        from clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
        from cluster_tracking.center.centers import get_centroids, get_medoids, get_modes, get_std, centers_df_to_dict
    else:
        from ...clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
        from ..center.centers import get_centroids, get_medoids, get_modes, get_std, centers_df_to_dict
else:
    from clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
    from cluster_tracking.center.centers import get_centroids, get_medoids, get_modes, get_std, centers_df_to_dict

def compute_cost_matrix(clusters1, clusters2, centroids1 = None, centroids2 = None, std1 = None, std2 = None,
                        lat_col='Latitude', long_col='Longitude', cluster_col='cluster_labs', distance_col='Distance', noise_label=-1,
                        distance_metric=haversine_np, alpha=1):
    if centroids1 is None: 
        centroids1 = centers_df_to_dict(
            get_centroids(clusters1, lat_col=lat_col, long_col=long_col, cluster_col=cluster_col, noise_label=noise_label))
    if centroids2 is None: 
        centroids2 = centers_df_to_dict(
            get_centroids(clusters2, lat_col=lat_col, long_col=long_col, cluster_col=cluster_col, noise_label=noise_label))
    if std1 is None: 
        std1 = centers_df_to_dict(
            get_std(clusters1, lat_col=lat_col, long_col=long_col, cluster_col=cluster_col, 
                    noise_label=noise_label, centers_df=centroids1, distance_col=distance_col, distance_metric=distance_metric))
    if std2 is None: 
        std2 = centers_df_to_dict(
            get_std(clusters2, lat_col=lat_col, long_col=long_col, cluster_col=cluster_col, 
                    noise_label=noise_label, centers_df=centroids2, distance_col=distance_col, distance_metric=distance_metric))

    num_clusters1 = len(centroids1.keys())
    num_clusters2 = len(centroids2.keys())
    
    # Initialize cost matrix with large values for dummy matches
    cost_matrix = np.full((num_clusters1, num_clusters2), np.inf)
    
    # Fill the actual cost matrix
    for i, c1 in enumerate(list(centroids1.keys())):
        for j, c2 in enumerate(list(centroids2.keys())):
            centroid_distance = distance_metric(centroids1[c1][long_col], centroids1[c1][lat_col], 
                                                centroids2[c2][long_col], centroids2[c2][lat_col])
            if centroid_distance < max(2*std1[c1][distance_col], 2*std2[c2][distance_col]):
                cost_matrix[i, j] = alpha*centroid_distance
    
    return cost_matrix

def match_clusters(clusters1, clusters2, lat_col='Latitude', long_col='Longitude', cluster_col='cluster_labs', distance_col='Distance', 
                   noise_label=-1, distance_metric=haversine_np, alpha=1):
    centroids1 = centers_df_to_dict(
        get_centroids(clusters1, lat_col=lat_col, long_col=long_col, cluster_col=cluster_col, noise_label=noise_label))
    centroids2 = centers_df_to_dict(
        get_centroids(clusters2, lat_col=lat_col, long_col=long_col, cluster_col=cluster_col, noise_label=noise_label))
    std1 = centers_df_to_dict(
        get_std(clusters1, lat_col=lat_col, long_col=long_col, cluster_col=cluster_col, 
                noise_label=noise_label, centers_df=centroids1, distance_col=distance_col, distance_metric=distance_metric))
    std2 = centers_df_to_dict(
        get_std(clusters2, lat_col=lat_col, long_col=long_col, cluster_col=cluster_col, 
                noise_label=noise_label, centers_df=centroids2, distance_col=distance_col, distance_metric=distance_metric))
    
    cluster1_ind = {c1: i for i, c1 in enumerate(centroids1.keys())}
    cluster2_ind = {c2: j for j, c2 in enumerate(centroids2.keys())}
    cost_matrix = compute_cost_matrix(clusters1, clusters2, centroids1, centroids2, std1, std2, 
                                      lat_col, long_col, cluster_col, distance_col, noise_label, distance_metric, alpha)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cluster1_ind, cluster2_ind, row_ind, col_ind