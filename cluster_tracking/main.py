import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import os
import sys
from center.centers import get_centroids, get_medoids, get_modes, get_std, centers_df_to_dict
if __name__=='__main__':
    if __package__ is None:
        import os.path as path
        sys.path.insert(0, path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
        from clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
    else:
        from ..clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
else:
    from clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE

def compute_cost_matrix(clusters1, clusters2, centroids1 = None, centroids2 = None, std1 = None, std2 = None,
                        lat_col='Latitude', long_col='Longitude', cluster_col='cluster_labs', distance_col='Distance', noise_label=-1,
                        distance_metric=haversine_np, alpha=1, beta=1):
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
    cost_matrix = np.full((num_clusters1 + num_clusters2, num_clusters1 + num_clusters2), np.inf)
    
    # Fill the actual cost matrix
    for i, c1 in enumerate(list(centroids1.keys())):
        for j, c2 in enumerate(list(centroids2.keys())):
            centroid_distance = distance_metric(centroids1[c1][long_col], centroids1[c1][lat_col], 
                                                centroids2[c2][long_col], centroids2[c2][lat_col])
            if centroid_distance < max(2*std1[c1][distance_col], 2*std2[c2][distance_col]):
                cost_matrix[i, j] = alpha*centroid_distance
    
    return cost_matrix

def main(data_path):
    df = pd.read_csv(data_path)
    df1 = df[(df['Year'] == 2008) & (df['Sector']=='U')]
    df2 = df[(df['Year'] == 2009) & (df['Sector']=='U')]

    cost_matrix = compute_cost_matrix(df1, df2)
    # print(cost_matrix)


if __name__=='__main__':
    data_path = 'clustering/test.csv'
    main(data_path)