import geopandas as gpd

# Input file paths
####### THIS IS A PATH WHERE I MANUALLY LABEL PLOTS WITH GROUND DATA ("Ground?" = "yes")
footprint_path = "/path/to/data.gpkg"


PD = 300
seedlings = 'all'    ###'big'

metrics_path = f"/path/to/data.gpkg"

output_path = metrics_path.replace('.gpkg', '_grID.gpkg')

# metrics_path = '/path/to/data.gpkg'
# output_path = "/path/to/data.gpkg"

# Load GeoPackages
footprint_gdf = gpd.read_file(footprint_path)
footprint_gdf = footprint_gdf[footprint_gdf.GroundID > 0]
metrics_gdf = gpd.read_file(metrics_path)

# Ensure CRS match
if metrics_gdf.crs != footprint_gdf.crs:
    footprint_gdf = footprint_gdf.to_crs(metrics_gdf.crs)

# Merge "Ground?" from footprint_gdf into metrics_gdf
metrics_with_ground = metrics_gdf.merge(
    footprint_gdf[["UniqueID", "plot_id", "GroundID"]],
    on=["UniqueID", "plot_id"],
    how="left"
)

# Rename "Ground?" to "GroundID" for clarity
# metrics_with_ground = metrics_with_ground.rename(columns={"Ground?": "GroundID"})

# Save the updated GeoDataFrame
metrics_with_ground.to_file(output_path, driver="GPKG")

print(f"Updated metrics with 'GroundID' saved to {output_path}")
