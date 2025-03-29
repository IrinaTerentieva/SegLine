import geopandas as gpd

# Input file paths
####### THIS IS A PATH WHERE I MANUALLY LABEL PLOTS WITH GROUND DATA ("Ground?" = "yes")
footprint_path = "file:///media/irina/data/Blueberry/DATA/PD300/16_trimmed_footprint_QC_plots20m2_noSlivers_newSegID_QC.gpkg"


PD = 300
seedlings = 'all'    ###'big'

metrics_path = f"file:///media/irina/data/Blueberry/DATA/status/all_sites_all_metrics_PD{PD}_v4.2_ass_{seedlings}seedlings.gpkg"

output_path = metrics_path.replace('.gpkg', '_grID.gpkg')

# metrics_path = '/media/irina/My Book/Recovery/DATA/vector_data/FLM_2024/Assessments/LiDea_2024_v5.4_plots100m2_metrics_v2_conf0.3.gpkg'
# output_path = "/media/irina/My Book/Recovery/DATA/vector_data/FLM_2024/Assessments/Plots/LiDea_2024_v5.4_plots100m2_metrics_conf0.3_with_groundid.gpkg"

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
