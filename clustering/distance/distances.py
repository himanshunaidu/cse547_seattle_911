import numpy as np

RADIUS_OF_EARTH_AT_SPACE_NEEDLE = 6366.512563943 # km

def meters_to_hav(meters, R=RADIUS_OF_EARTH_AT_SPACE_NEEDLE):
    """Converts a distance in meters to haversine distance"""
    hav = meters / (R * 1000)
    return hav

def haversine_np(lon1, lat1, lon2, lat2, R=RADIUS_OF_EARTH_AT_SPACE_NEEDLE):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    return km

def linearized_haversine(a, b, R=RADIUS_OF_EARTH_AT_SPACE_NEEDLE):
    a_rad = a * (2 * np.pi / 360)
    b_rad = b * (2 * np.pi / 360)
    x_1, y_1 = a_rad
    x_2, y_2 = b_rad
    delta_x = R * np.cos(y_1) * (x_2 - x_1)
    delta_y = R * (y_2 - y_1)
    return np.sqrt(delta_x**2 + delta_y**2)