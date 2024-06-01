import numpy as np
import pandas as pd
from pyemd import emd

def calculate_partial_emd(cluster1, cluster2):
    # Combine clusters into a single dataset for partial matching
    combined_cluster = np.vstack((cluster1, cluster2))
    num_points_1 = cluster1.shape[0]
    num_points_2 = cluster2.shape[0]
    
    # Create signatures with weights
    signature1 = np.hstack((cluster1, np.ones((num_points_1, 1)) / num_points_1))
    print(signature1[:, -1])
    return
    signature2 = np.hstack((cluster2, np.ones((num_points_2, 1)) / num_points_2))
    
    # Distance matrix
    distance_matrix = np.linalg.norm(signature1[:, None, :-1] - signature2[None, :, :-1], axis=2)
    
    # Calculate EMD with partial matching
    emd_distance = emd(signature1[:, -1], signature2[:, -1], distance_matrix)
    return emd_distance

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

    # for key in freq_set1.difference(freq_set2): freq_dict2[key] = 0
    # for key in freq_set2.difference(freq_set1): freq_dict1[key] = 0
    locations = {i: location for i, location in enumerate(freq_set1.union(freq_set2))}
    for location in locations.values():
        if location not in freq_dict1: freq_dict1[location] = 0
        if location not in freq_dict2: freq_dict2[location] = 0

    hist1 = np.array([freq_dict1[location] for location in locations.values()])
    hist2 = np.array([freq_dict2[location] for location in locations.values()])
    return hist1, hist2, locations

def main(data_path):
    df = pd.read_csv(data_path)
    df1 = df[(df['Year'] == 2008) & (df['Sector']=='U') & (df['cluster_labs']==0)]
    df2 = df[(df['Year'] == 2009) & (df['Sector']=='U') & (df['cluster_labs']==0)]

    # print(df1.shape, df2.shape)
    hist1, hist2, locations = create_histogram_maps(df1, df2)
    print(locations)

if __name__=="__main__":
    data_path = 'clustering/test.csv'
    main(data_path)