import geopandas as gpd

def extract_zone_centroids(shapefile_path: str, output_path: str):
    gdf = gpd.read_file(shapefile_path)
    gdf["centroid"] = gdf.geometry.centroid
    gdf["lat"] = gdf.centroid.y
    gdf["lon"] = gdf.centroid.x

    zone_coords = gdf[["LocationID", "lat", "lon"]].sort_values("LocationID")
    zone_coords.to_csv(output_path, index=False)
    print(f"Saved zone coordinates to {output_path}")

if __name__ == "__main__":
    shapefile = "data/raw/taxi_zones/taxi_zones.shp"
    output = "data/processed/zone_coords.csv"
    extract_zone_centroids(shapefile, output)