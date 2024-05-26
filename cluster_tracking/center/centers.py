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


def get_centroids(df: pd.DataFrame, lat_col: str='lat_rad', long_col: str='long_rad', 
                  cluster_label: str='cluster_labs', noise_label: int|str=-1):
    if lat_col not in df.columns or long_col not in df.columns:
        raise ValueError('Latitude or longitude columns not found in dataframe')
    if cluster_label not in df.columns:
        raise ValueError('Cluster labels not found in dataframe')
    
    centroids = df[df[cluster_label]!=noise_label].groupby(cluster_label).agg({lat_col: 'mean', long_col: 'mean'}).reset_index()
    return centroids

if __name__=='__main__':
    df = pd.read_csv('clustering/test.csv')
    df = df[df['Year'] == 2008]
    centroids = get_centroids(df)
    print(centroids.shape)
    centroids['lat'] = np.degrees(centroids['lat_rad'])
    centroids['long'] = np.degrees(centroids['long_rad'])
    color_scale = [(0, 'orange'), (1,'red')]
    fig = px.scatter_mapbox(centroids, 
                        lat="lat", 
                        lon="long", 
                        # hover_name="Address", 
                        # hover_data=["Address", "Listed"],
                        # color="Listed",
                        color_continuous_scale=color_scale,
                        # size="Listed",
                        zoom=8, 
                        height=800,
                        width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # plt.plot(centroids['lat'], centroids['long'], 'ro')
    fig.show()
