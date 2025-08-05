import geopandas as gpd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

def load_zone_centroids(shapefile_path: Path) -> dict[int, tuple[float, float]]:
    gdf = gpd.read_file(shapefile_path)

    gdf_proj = gdf.to_crs("EPSG:2263")  
    gdf["centroid"] = gdf_proj.geometry.centroid.to_crs("EPSG:4326")
    gdf["lat"] = gdf["centroid"].y
    gdf["lon"] = gdf["centroid"].x

    zone_to_coords = dict(zip(gdf["LocationID"], zip(gdf["lat"], gdf["lon"])))
    return zone_to_coords


def compute_zone_distance_matrix(zone_to_coords: dict[int, tuple[float, float]]) -> pd.DataFrame:
    data = []
    zone_ids = list(zone_to_coords.keys())

    for pu in zone_ids:
        for do in zone_ids:
            lat1, lon1 = zone_to_coords[pu]
            lat2, lon2 = zone_to_coords[do]
            km = haversine_distance(lat1, lon1, lat2, lon2)
            data.append((pu, do, km))
    
    df = pd.DataFrame(data, columns=["PULocationID", "DOLocationID", "great_circle_km"])
    return df


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


if __name__ == "__main__":
    path = Path("data/geo/taxi_zones/taxi_zones.shp")
    zone_centroids = load_zone_centroids(path)

    print(f"Loaded {len(zone_centroids)} zone centroids.")
    print("Example:")
    for k, v in list(zone_centroids.items())[:5]:
        print(f"Zone {k}: lat={v[0]:.5f}, lon={v[1]:.5f}")
    distance_df = compute_zone_distance_matrix(zone_centroids)
    print(distance_df.head())

    # Save to file
    distance_df.to_parquet("data/geo/zone_distance_matrix.parquet", index=False)
    print("✅ Saved distance matrix to data/geo/zone_distance_matrix.parquet")