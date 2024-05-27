import math
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from collections.abc import Callable

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

def arg_check(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', 
                  cluster_col: str='cluster_labs', noise_label: int|str=-1) -> None:
    if lat_col not in df.columns or long_col not in df.columns:
        raise ValueError('Latitude or longitude columns not found in dataframe')
    if cluster_col not in df.columns:
        raise ValueError('Cluster labels not found in dataframe')

# Not affected by distance metric
def get_centroids(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', 
                  cluster_col: str='cluster_labs', noise_label: int|str=-1):
    arg_check(df, lat_col, long_col, cluster_col, noise_label)
    
    gb = df[df[cluster_col]!=noise_label].groupby(cluster_col)
    centroids = gb.agg({lat_col: 'mean', long_col: 'mean'}).reset_index()
    # centroids = centroids.join(gb.size().to_frame(name='counts')).reset_index()
    return centroids

# Not affected by distance metric
def get_medoids(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', 
                cluster_col: str='cluster_labs', noise_label: int|str=-1):
    arg_check(df, lat_col, long_col, cluster_col, noise_label)
    
    gb = df[df[cluster_col]!=noise_label].groupby(cluster_col)
    medoids = gb.agg({lat_col: 'median', long_col: 'median'}).reset_index()
    # medoids = medoids.join(gb.size().to_frame(name='counts')).reset_index()
    return medoids

def get_modes(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', 
              cluster_col: str='cluster_labs', noise_label: int|str=-1):
    arg_check(df, lat_col, long_col, cluster_col, noise_label)
    
    # modes = df[df[cluster_col]!=noise_label].groupby(cluster_col).agg({lat_col: lambda x: x.value_counts().idxmax(), 
    #                                                                      long_col: lambda x: x.value_counts().idxmax()}).reset_index()
    gb = df[df[cluster_col]!=noise_label].groupby(cluster_col)
    modes = gb.agg({lat_col: pd.Series.mode, long_col: pd.Series.mode}).reset_index()
    # modes = modes.join(gb.size().to_frame(name='counts')).reset_index()
    return modes

def get_std(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', 
              cluster_col: str='cluster_labs', noise_label: int|str=-1, 
              distance_col: str='Distance', centers_df: pd.DataFrame=None, distance_metric: Callable=haversine_np):
    arg_check(df, lat_col, long_col, cluster_col, noise_label)
    if not centers_df: centers_df = centers_df_to_dict(get_centroids(df, lat_col, long_col, cluster_col, noise_label))

    df_filtered = df[df[cluster_col]!=noise_label]

    df_filtered[distance_col] = df_filtered.apply(
        lambda row: distance_metric(row[long_col], row[lat_col], 
                                 centers_df[row[cluster_col]][long_col], centers_df[row[cluster_col]][lat_col]), axis=1)
    
    gb = df_filtered.groupby(cluster_col)
    variances = gb.agg({distance_col: np.std, lat_col: 'mean', long_col: 'mean'}).reset_index()
    # modes = modes.join(gb.size().to_frame(name='counts')).reset_index()
    return variances

def centers_df_to_dict(centers_df: pd.DataFrame, cluster_col: str='cluster_labs', 
                       lat_col: str='Latitude', long_col: str='Longitude'):
    centers_df = centers_df.set_index(cluster_col)
    return centers_df.to_dict(orient='index')

if __name__=='__main__':
    df = pd.read_csv('clustering/test.csv')
    df = df[(df['Year'] == 2008) & (df['cluster_labs']!=-1) & (df['Sector']=='U')]
    print(df['Year'].unique(), df['cluster_labs'].unique(), df['Sector'].unique())
    centroids = get_std(df)
    print(centroids)
    # print(centroids.apply(lambda row: math.sqrt(row['Distance']), axis=1))
    print(centers_df_to_dict(centroids))
    # centroids['lat'] = np.degrees(centroids['lat_rad'])
    # centroids['long'] = np.degrees(centroids['long_rad'])
    color_scale = [(0, 'orange'), (1,'red')]
    fig = px.scatter_mapbox(centroids, 
                        lat="Latitude", 
                        lon="Longitude", 
                        hover_name="cluster_labs", 
                        hover_data=["cluster_labs", "Distance"],
                        color="cluster_labs",
                        # color_continuous_scale=color_scale,
                        size="Distance",
                        zoom=10, 
                        height=800,
                        width=800)
    
    # Plot individual points 
    # fig = px.scatter_mapbox(df, 
    #                     lat="Latitude", 
    #                     lon="Longitude", 
    #                     hover_name="cluster_labs", 
    #                     hover_data=["cluster_labs"],
    #                     color="cluster_labs",
    #                     # color_discrete_sequence=color_scale,
    #                     # size="counts",
    #                     zoom=10, 
    #                     height=800,
    #                     width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # plt.plot(centroids['lat'], centroids['long'], 'ro')
    fig.show()
