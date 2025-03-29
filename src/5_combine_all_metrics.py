import geopandas as gpd
import os
import glob
import re
import pandas as pd

def merge_metrics_for_all_sites(metrics_folder):
    """
    Merges metrics1, metrics2, and metrics3 files for each unique site in the given folder,

    :param metrics_folder: Path to the folder containing the metric files.
    """
    # Find all metric files
    metrics_files = glob.glob(os.path.join(metrics_folder, "*_metrics[1-3]_datasets_PD*.gpkg"))

    # Extract unique site names and PD values
    site_pd_map = {}
    for file in metrics_files:
        match = re.search(r"(.*?)_metrics[1-3]_datasets_(PD\d+)\.gpkg", os.path.basename(file))
        if match:
            site_name, pd_value = match.groups()
            site_pd_map.setdefault((site_name, pd_value), []).append(file)

    for (site, pd_value), site_metrics_files in site_pd_map.items():
        print(f"Processing site: {site} for {pd_value}")

        # Ensure we have exactly 3 metric files
        if len(site_metrics_files) != 3:
            print(f"Skipping {site} ({pd_value}): Expected 3 metric files but found {len(site_metrics_files)}")
            continue

        site_metrics_files = sorted(site_metrics_files)  # Ensure correct order

        # Read metrics1
        metrics1 = gpd.read_file(site_metrics_files[0])

        # Read and join metrics2
        metrics2 = gpd.read_file(site_metrics_files[1])
        cols_to_drop = [col for col in metrics2.columns if col in metrics1.columns and col not in ["UniqueID", "SegmentID", "plot_id"]]
        metrics2 = metrics2.drop(columns=cols_to_drop)
        merged_data = metrics1.merge(metrics2, on=["UniqueID", "SegmentID", "plot_id"], how="left")

        # Read and join metrics3
        metrics3 = gpd.read_file(site_metrics_files[2])
        cols_to_drop = [col for col in metrics3.columns if col in merged_data.columns and col not in ["UniqueID", "SegmentID", "plot_id"]]
        metrics3 = metrics3.drop(columns=cols_to_drop)
        merged_data = merged_data.merge(metrics3, on=["UniqueID", "SegmentID", "plot_id"], how="left")

        # Move 'sitename' to be the first column if it exists
        if "sitename" in merged_data.columns:
            cols = ["sitename"] + [col for col in merged_data.columns if col != "sitename"]
            merged_data = merged_data[cols]

        # Define output path
        output_path = os.path.join(metrics_folder, f"{site}_all_metrics_{pd_value}.gpkg")

        # Save the merged dataset
        merged_data.to_file(output_path, driver="GPKG")

        print(f"Successfully merged metrics for {site} with {pd_value} and saved to: {output_path}")


def merge_all_sites(metrics_folder):
    """
    Merges all site-specific all_metrics files into a single dataset per PD value.

    :param metrics_folder: Path to the folder containing the all_metrics files.
    """
    # Find all site-specific all_metrics files
    all_metrics_files = glob.glob(os.path.join(metrics_folder, "*_all_metrics_PD*.gpkg"))

    # Extract unique PD values
    pd_values = set(re.search(r"PD\d+", f).group() for f in all_metrics_files if re.search(r"PD\d+", f))

    for pd_value in pd_values:
        print(f"Merging all sites for {pd_value}")

        # Select files for the current PD value
        pd_files = [f for f in all_metrics_files if pd_value in f]

        # Read and concatenate all site datasets
        merged_gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in pd_files], ignore_index=True))

        # Define output path
        output_path = os.path.join(metrics_folder, f"all_sites_all_metrics_{pd_value}.gpkg")

        # Save the merged dataset
        merged_gdf.to_file(output_path, driver="GPKG")

        print(f"Successfully merged all sites for {pd_value} and saved to: {output_path}")

# merge per PD
# metrics_folder_PD300 = r"D:\Blueberry_NEW\7_products\PD300\metrics"
# merge_metrics_for_all_sites(metrics_folder_PD300)
# merge_all_sites(metrics_folder_PD300)

metrics_folder_PD25 = r"D:\Blueberry_NEW\7_products\PD25\metrics"
merge_metrics_for_all_sites(metrics_folder_PD25)
merge_all_sites(metrics_folder_PD25)

metrics_folder_PD5 = r"D:\Blueberry_NEW\7_products\PD5\metrics"
merge_metrics_for_all_sites(metrics_folder_PD5)
merge_all_sites(metrics_folder_PD5)
