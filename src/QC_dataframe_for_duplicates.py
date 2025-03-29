import geopandas as gpd
import os

# Define input paths
for PD in ['PD300', 'PD25', 'PD5']:
    plots_gpkg = f"/media/irina/data/Blueberry/DATA/metrics/all_sites_all_metrics_{PD}_v4.2.gpkg"

    # Load the GeoPackage
    if not os.path.exists(plots_gpkg):
        print(f"❌ File not found: {plots_gpkg}")
        continue

    gdf = gpd.read_file(plots_gpkg)
    print('Len of original: ', len(gdf))

    # Check for duplicates based on 'plot_id'
    duplicate_rows = gdf[gdf.duplicated(subset=['plot_id'], keep=False)]

    if not duplicate_rows.empty:
        print(f"⚠️ Found {len(duplicate_rows)} duplicate rows in {PD}. Removing duplicates...")

        # Keep only the first occurrence of each duplicate plot_id
        gdf = gdf.drop_duplicates(subset=['plot_id'], keep='first')

        # Save the cleaned dataset
        cleaned_gpkg = f"/media/irina/data/Blueberry/DATA/metrics/all_sites_all_metrics_{PD}_v4.2_cleaned.gpkg"
        print('Len of cleaned: ', len(gdf))
        gdf.to_file(cleaned_gpkg, driver="GPKG")
        print(f"✅ Cleaned dataset saved: {cleaned_gpkg}")
    else:
        print(f"✅ No duplicates found in {PD}.")
