from collections import defaultdict
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Load the HOLC GeoJSON file
holc_fp = "cluster_tracking/redlining/redlining_seattle.json"
all_redlining_value_counts_path = 'cluster_tracking/redlining/all_redlining_value_counts.csv'
grade_keys = ['A', 'B', 'C', 'D', 'None']

def get_holc_grade(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', redlining_data_path: str=holc_fp):
    holc_data = gpd.read_file(redlining_data_path)
    points = df.apply(lambda row: Point(row[long_col], row[lat_col]), axis=1)

    gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    gdf_points = gdf_points.to_crs(holc_data.crs)

    holc_df = gpd.sjoin(gdf_points, holc_data, how="left", predicate="within")
    holc_df_grade = pd.DataFrame(holc_df[['geometry', 'grade']])
    return holc_df_grade

# def get_yearly_holc_value_counts(df: pd.DataFrame, lat_col: str='Latitude', long_col: str='Longitude', year_col: str='Year', 
#                                       redlining_data_path: str=holc_fp):
#     yearly_holc_value_counts = pd.DataFrame(columns=[year_col, 'grade', 'count'])
#     years = df[year_col].unique()
#     for year in years:
#         df_year = df[df[year_col] == year]
#         holc_df = get_holc_grade(df_year, lat_col, long_col, redlining_data_path)
#         holc_value_counts = holc_df['grade'].value_counts(dropna=False)
#         yearly_holc_value_counts = pd.concat([yearly_holc_value_counts, holc_value_counts], axis=0)
#         print(f'Year: {year}')
#     print(yearly_holc_value_counts.head())

def get_all_redlining_value_counts():
    return pd.read_csv(all_redlining_value_counts_path, index_col='grade')['count']

def holc_grade_counts_to_dict(holc_df_counts: pd.Series):
    holc_counts_dict = defaultdict(int)
    for index, row in holc_df_counts.items():
        row_value = max(row, 5)
        row_grade = 'None' if isinstance(index, type(None)) or isinstance(index, float) else index
        if row_grade in holc_counts_dict:
            holc_counts_dict[row_grade] += row_value
        else:
            holc_counts_dict[row_grade] = row_value
    for grade in grade_keys:
        if grade not in holc_counts_dict:
            holc_counts_dict[grade] = 5
    return holc_counts_dict

def normalize_count_dict(count_dict):
    total = sum(count_dict.values())
    return {key: value*100/total for key, value in count_dict.items()}

def run_chi_square_test(holc_counts_dict1, holc_counts_dict2):
    observed = [holc_counts_dict1[grade] for grade in grade_keys]
    expected = [holc_counts_dict2[grade] for grade in grade_keys]
    chi2, p = chisquare(observed, expected)
    return chi2, p


if __name__ == '__main__':
    df = pd.read_csv('clustering/test.csv')
    holc_df = get_holc_grade(df)
    holc_value_counts = holc_df['grade'].value_counts(dropna=False)
    # print(pd.concat([holc_value_counts, holc_value_counts], axis=1).head())
    holc_value_counts.to_csv(all_redlining_value_counts_path, index=False)
    # get_yearly_holc_value_counts(df)