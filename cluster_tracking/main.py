import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import os
import sys
from center.centers import get_centroids, get_medoids, get_modes, get_std, centers_df_to_dict
from match.hungarian import match_clusters
from distance.emd import get_emd
if __name__=='__main__':
    if __package__ is None:
        import os.path as path
        sys.path.insert(0, path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
        from clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
    else:
        from ..clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
else:
    from clustering.distance.distances import haversine_np, meters_to_hav, RADIUS_OF_EARTH_AT_SPACE_NEEDLE


def main(data_path):
    df = pd.read_csv(data_path)
    df1 = df[(df['Year'] == 2008) & (df['Sector']=='U')]
    df2 = df[(df['Year'] == 2009) & (df['Sector']=='U')]

    cluster1_ind, cluster2_ind, row_ind, col_ind = match_clusters(df1, df2)
    ind1_cluster1 = {v: k for k, v in cluster1_ind.items()}
    ind2_cluster2 = {v: k for k, v in cluster2_ind.items()}
    # print(cluster1_ind, cluster2_ind)
    # print(row_ind, col_ind)
    cluster1_no_match = set(cluster1_ind.keys()).difference(set(ind1_cluster1[i] for i in row_ind))
    cluster2_no_match = set(cluster2_ind.keys()).difference(set(ind2_cluster2[j] for j in col_ind))
    print(cluster1_no_match, cluster2_no_match)
    
    distance = 0
    for row, col in zip(row_ind, col_ind):
        distance += get_emd(df1[df1['cluster_labs']==ind1_cluster1[row]], df2[df2['cluster_labs']==ind2_cluster2[col]])
    for c1 in cluster1_no_match:
        distance += get_emd(df1[df1['cluster_labs']==c1], df2[df2['cluster_labs']==-1])
    for c2 in cluster2_no_match:
        distance += get_emd(df1[df1['cluster_labs']==-1], df2[df2['cluster_labs']==c2])
    print(distance)
    print(distance/(df1.shape[0]+df2.shape[0]))


if __name__=='__main__':
    data_path = 'clustering/test.csv'
    main(data_path)