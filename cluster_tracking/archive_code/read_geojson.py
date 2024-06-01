import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Load the HOLC GeoJSON file
holc_fp = "cluster_tracking\\redlining\\redlining_seattle.json"  # Replace with your GeoJSON file path
holc_data = gpd.read_file(holc_fp)
# holc_data.plot()
# plt.show()

# Define your coordinates (example)
coordinates = [(47.6095, -122.3335), (47.6205, -122.3493), (47.61306, -122.29277)]  # Add your coordinates here

# Create a GeoDataFrame from the coordinates
points = [Point(lon, lat) for lat, lon in coordinates]
gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

# Reproject the points to match the HOLC GeoJSON CRS
gdf_points = gdf_points.to_crs(holc_data.crs)

# Spatial join to find HOLC grades for the points
joined = gpd.sjoin(gdf_points, holc_data, how="left", predicate="within")

# Display the results
# print(type(joined[['geometry', 'grade']]))
joined_df = pd.DataFrame(joined[['geometry', 'grade']])
joined_df['point'] = joined_df['geometry'].apply(lambda x: x.coords[0])
print(joined_df)
print(joined_df.dtypes)