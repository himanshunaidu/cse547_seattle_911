import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import os
import sys
if __name__=='__main__':
    if __package__ is None:
        import os.path as path
        sys.path.insert(0, path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
        from clustering.distance.distances import haversine_np, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
    else:
        from ...clustering.distance.distances import haversine_np, RADIUS_OF_EARTH_AT_SPACE_NEEDLE
else:
    from clustering.distance.distances import haversine_np, RADIUS_OF_EARTH_AT_SPACE_NEEDLE

# Not affected by distance metric
def get_centroids(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', 
                  cluster_label: str='cluster_labs', noise_label: int|str=-1):
    if lat_col not in df.columns or long_col not in df.columns:
        raise ValueError('Latitude or longitude columns not found in dataframe')
    if cluster_label not in df.columns:
        raise ValueError('Cluster labels not found in dataframe')
    
    gb = df[df[cluster_label]!=noise_label].groupby(cluster_label)
    centroids = gb.agg({lat_col: 'mean', long_col: 'mean'})
    centroids = centroids.join(gb.size().to_frame(name='counts')).reset_index()
    return centroids

# Not affected by distance metric
def get_medoids(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', 
                cluster_label: str='cluster_labs', noise_label: int|str=-1):
    if lat_col not in df.columns or long_col not in df.columns:
        raise ValueError('Latitude or longitude columns not found in dataframe')
    if cluster_label not in df.columns:
        raise ValueError('Cluster labels not found in dataframe')
    
    gb = df[df[cluster_label]!=noise_label].groupby(cluster_label)
    medoids = gb.agg({lat_col: 'median', long_col: 'median'})
    medoids = medoids.join(gb.size().to_frame(name='counts')).reset_index()
    return medoids

def get_modes(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', 
              cluster_label: str='cluster_labs', noise_label: int|str=-1):
    if lat_col not in df.columns or long_col not in df.columns:
        raise ValueError('Latitude or longitude columns not found in dataframe')
    if cluster_label not in df.columns:
        raise ValueError('Cluster labels not found in dataframe')
    
    # modes = df[df[cluster_label]!=noise_label].groupby(cluster_label).agg({lat_col: lambda x: x.value_counts().idxmax(), 
    #                                                                      long_col: lambda x: x.value_counts().idxmax()}).reset_index()
    gb = df[df[cluster_label]!=noise_label].groupby(cluster_label)
    modes = gb.agg({lat_col: pd.Series.mode, long_col: pd.Series.mode})
    modes = modes.join(gb.size().to_frame(name='counts')).reset_index()
    return modes

if __name__=='__main__':
    df = pd.read_csv('clustering/test.csv')
    df = df[(df['Year'] == 2008) & (df['cluster_labs']!=-1) & (df['Sector']=='U')]
    print(df['Year'].unique(), df['cluster_labs'].unique(), df['Sector'].unique())
    centroids = get_centroids(df)
    # centroids['lat'] = np.degrees(centroids['lat_rad'])
    # centroids['long'] = np.degrees(centroids['long_rad'])
    color_scale = [(0, 'orange'), (1,'red')]
    fig = px.scatter_mapbox(centroids, 
                        lat="Latitude", 
                        lon="Longitude", 
                        hover_name="cluster_labs", 
                        hover_data=["cluster_labs", "counts"],
                        color="cluster_labs",
                        # color_continuous_scale=color_scale,
                        size="counts",
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
