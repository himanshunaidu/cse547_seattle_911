import numpy as np
import pandas as pd
from pyemd import emd

import os
import os.path as path
import sys

if __name__=='__main__':
    if __package__ is None:
        sys.path.insert(0, path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
        from clustering.distance.distances import haversine_np, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
    else:
        from ...clustering.distance.distances import haversine_np, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
else:
    sys.path.insert(0, path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
    from clustering.distance.distances import haversine_np, RADIUS_OF_EARTH_AT_SPACE_NEEDLE

def create_histogram_maps(df1: pd.DataFrame, df2: pd.DataFrame, lat_col='Latitude', long_col='Longitude', 
                        cluster_col='cluster_labs', distance_col='Distance', noise_label=-1, frequency_col='Frequency'):
    df_filtered1 = df1[df1[cluster_col]!=noise_label]
    df_filtered2 = df2[df2[cluster_col]!=noise_label]
    
    freq1 = df_filtered1.groupby(by=[lat_col, long_col]).size().reset_index(name=frequency_col)
    # freq_dict1 = freq1[["Latitude", "Longitude"]].to_dict(orient='index')
    freq_dict1 = { (row[lat_col], row[long_col]): row[frequency_col] for _, row in freq1.iterrows() }
    freq_set1 = set(freq_dict1.keys())

    freq2 = df_filtered2.groupby(by=[lat_col, long_col]).size().reset_index(name="Frequency")
    freq_dict2 = { (row[lat_col], row[long_col]): row[frequency_col] for _, row in freq2.iterrows() }
    freq_set2 = set(freq_dict2.keys())

    locations = {i: location for i, location in enumerate(freq_set1.union(freq_set2))}
    for location in locations.values():
        if location not in freq_dict1: freq_dict1[location] = 0.0
        if location not in freq_dict2: freq_dict2[location] = 0.0

    hist1 = np.array([freq_dict1[location] for location in locations.values()])
    hist2 = np.array([freq_dict2[location] for location in locations.values()])
    return hist1, hist2, locations

def calculate_distance_matrix(hist1: np.ndarray, hist2: np.ndarray, locations: dict):
    distance_matrix = np.zeros((len(locations.keys()), len(locations.keys())))
    
    for i in locations.keys():
        for j in locations.keys():
            distance_matrix[i, j] = haversine_np(locations[i][1], locations[i][0], locations[j][1], locations[j][0])
    return distance_matrix


def get_emd(df1: pd.DataFrame, df2: pd.DataFrame, lat_col='Latitude', long_col='Longitude', 
                        cluster_col='cluster_labs', distance_col='Distance', noise_label=-1):
    hist1, hist2, locations = create_histogram_maps(df1, df2, lat_col, long_col, cluster_col, distance_col, noise_label)
    distance_matrix = calculate_distance_matrix(hist1, hist2, locations)

    emd_distance = emd(hist1, hist2, distance_matrix)
    return emd_distance


if __name__=="__main__":
    data_path = 'clustering/test.csv'
    df = pd.read_csv(data_path)
    df1 = df[(df['Year'] == 2008) & (df['Sector']=='U') & (df['cluster_labs']==0)]
    df2 = df[(df['Year'] == 2009) & (df['Sector']=='U') & (df['cluster_labs']==100)]

    get_emd(df1, df2)