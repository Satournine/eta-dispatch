import pandas as pd

import geopandas as gpd


def load_zone_latlons(path="data/processed/zone_coords.csv"):
    df = pd.read_csv(path)
    zone_coords = {
        row["LocationID"]: (row["lat"], row["lon"])
        for _, row in df.iterrows()
    }
    return zone_coords

def convert_zone_centroids(input_path="data/raw//taxi_zones/taxi_zones.shp", output_path="data/processed/zone_coords.csv"):
    gdf = gpd.read_file(input_path)
    gdf = gdf.to_crs(epsg=4326)  # Convert to GPS (lat/lon)
    gdf["lon"] = gdf.geometry.centroid.map(lambda point: point.x)
    gdf["lat"] = gdf.geometry.centroid.map(lambda point: point.y)
    gdf[["LocationID", "lat", "lon"]].to_csv(output_path, index=False)

if __name__ == "__main__":
    convert_zone_centroids()