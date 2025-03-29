import geopandas as gpd
import os
import pandas as pd

# Define input folder and output file
input_folder = "/media/irina/data/Blueberry/DATA/shadows/all_sites_all_metrics_PD300_shadows.gpkg"
output_gpkg = os.path.join(input_folder, "BRFN_shadows.gpkg")

# Find all .gpkg files in the folder
gpkg_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".gpkg")]

if not gpkg_files:
    print("❌ No GPKG files found in the directory.")
    exit()

# Read all GPKG files and store in a list
gdfs = []
for gpkg in gpkg_files:
    print(f"📂 Reading: {gpkg}")
    gdf = gpd.read_file(gpkg)

    # ✅ Drop exact duplicate rows (if any)
    gdf = gdf.drop_duplicates()

    gdfs.append(gdf)

# Merge all GeoDataFrames
merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

# Ensure CRS consistency
crs = gdfs[0].crs  # Assume all files have the same CRS
merged_gdf = merged_gdf.set_crs(crs)

# ✅ Drop duplicate `UniqueID, SegmentID` (keep the first occurrence)
merged_gdf = merged_gdf.drop_duplicates(subset=["UniqueID", "SegmentID"])

# Save merged GeoDataFrame
merged_gdf.to_file(output_gpkg, driver="GPKG")
print(f"✅ Merged GPKG saved: {output_gpkg}")



for PD in ['PD300', 'PD25', 'PD5']:

    # File paths
    metrics_file = f"/media/irina/data/Blueberry/DATA/metrics/all_sites_all_metrics_{PD}_v4.1.gpkg"

    # Load the datasets
    print("Loading metrics data...")
    metrics_gdf = gpd.read_file(metrics_file)

    # ✅ Drop duplicate `UniqueID, SegmentID` from metrics before merging
    metrics_gdf = metrics_gdf.drop_duplicates(subset=["UniqueID", "plot_id"])

    # ✅ Merge based on 'UniqueID' and 'SegmentID' (Left join to prevent duplication)
    print("Merging datasets...")
    with_shadows_gdf = metrics_gdf.merge(
        merged_gdf[["UniqueID", "plot_id", "shadow_coverage"]],
        on=["UniqueID", "plot_id"],
        how="left"
    )

    # ✅ Double-check for unexpected duplicates after merging
    duplicate_check = with_shadows_gdf.duplicated(subset=["UniqueID", "SegmentID"], keep=False).sum()
    if duplicate_check > 0:
        print(f"⚠️ Warning: {duplicate_check} duplicate rows detected after merging!")

    # Save back to the metrics file
    output_file = metrics_file.replace(".1.gpkg", ".2.gpkg")  # Avoid overwriting original
    print(f"Saving merged data to {output_file}...")
    with_shadows_gdf.to_file(output_file, driver="GPKG")

    print("✅ Merge complete!")
