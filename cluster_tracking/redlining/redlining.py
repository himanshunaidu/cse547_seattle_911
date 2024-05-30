import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Load the HOLC GeoJSON file
holc_fp = "cluster_tracking/redlining/redlining_seattle.json"

def get_holc_grade(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', redlining_data_path: str=holc_fp):
    holc_data = gpd.read_file(redlining_data_path)
    points = df.apply(lambda row: Point(row[long_col], row[lat_col]), axis=1)

    gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    gdf_points = gdf_points.to_crs(holc_data.crs)

    joined = gpd.sjoin(gdf_points, holc_data, how="left", predicate="within")
    joined_df = pd.DataFrame(joined[['geometry', 'grade']])
    return joined_df

if __name__ == '__main__':
    df = pd.read_csv('clustering/test.csv')
    get_holc_grade(df)