import geopandas as gpd
import pandas as pd
from rasterio.features import geometry_mask
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import yaml
import platform

# parallelization works for this script
# also using spatial indexing


def count_adjacency(seedlings_gdf, buffer_polygon):
    """
    Debug adjacency counts and density.
    """
    adjacency_seedlings = seedlings_gdf[seedlings_gdf.geometry.within(buffer_polygon)]

    total_seedlings = len(adjacency_seedlings)
    adjacency_class_counts = adjacency_seedlings["class"].value_counts().to_dict()

    # print("Adjacency counts:")
    # print(f"Total seedlings in buffer: {total_seedlings}")
    # print(f"Class counts: {adjacency_class_counts}")

    return total_seedlings, adjacency_class_counts


def calculate_side_coverage(segment_group, seedlings_gdf, wide=False):
    """
    Calculate side coverage (percentage of plots with seedlings) for each side of a segment.
    For wide lines, a plot is considered "good" if it has at least 2 seedlings,
    and a plot with 1 seedling counts as 0.5.

    """
    total_plots = segment_group["plot_id"].nunique()
    side_coverage = {}

    # Iterate through sides (e.g., 0 and 1)
    for side in [0, 1]:
        # Filter plots belonging to the current side
        side_plots = segment_group[segment_group["side"] == side]

        # Get unique plot IDs for this side
        total_side_plots = side_plots["plot_id"].nunique()

        # Seedlings in plots for this side
        seedlings_in_side = seedlings_gdf[seedlings_gdf["plot_id"].isin(side_plots["plot_id"])]

        if wide:
            # Group seedlings by plot_id and count seedlings
            seedling_counts = seedlings_in_side.groupby("plot_id").size()

            # Calculate "good plots" for wide lines (1+ as 0.5, 2+ as 1)
            good_plots = 0
            for count in seedling_counts:
                if count >= 2:
                    good_plots += 1  # Count as 1 for 2+ seedlings
                elif count == 1:
                    good_plots += 0.5  # Count as 0.5 for 1 seedling
        else:
            # For non-wide lines, consider plots with at least 1 seedling
            good_plots = len(seedlings_in_side["plot_id"].unique())

        # Calculate coverage for this side
        side_coverage[f"side{side}_coverage"] = (
            (good_plots / total_side_plots) * 100 if total_side_plots > 0 else 0
        )

    return side_coverage

def calculate_stocking(seedling_counts):
    stocking = 0
    for count in seedling_counts:
        if count >= 2:
            stocking += 1  # Count as 1 for 2+ seedlings
        elif count == 1:
            stocking += 0.5  # Count as 0.5 for 1 seedling
    return stocking

def calculate_metrics_for_segments(unique_id, segment_id, segment_group, polygon_gdf,
                                   seedlings_gdf, adjacency_buffer, adjacency_gap_buffer):
    """
    Calculate metrics for each segment in the unified dataset.
    """
    results = []

    # ✅ Create adjacency area
    combined_polygon = segment_group.geometry.union_all()
    wide = segment_group.head(1).geometry.iloc[0].area > 15

    buffer_polygon = combined_polygon.buffer(adjacency_buffer).difference(combined_polygon.buffer(adjacency_gap_buffer))

    # ✅ Also remove nearby segments
    nearby_polygons = polygon_gdf[polygon_gdf.intersects(buffer_polygon)]
    buffer_polygon = buffer_polygon.difference(nearby_polygons.buffer(adjacency_gap_buffer).union_all())

    # ✅ Detect adjacency counts
    total_seedlings, adjacency_class_counts = count_adjacency(seedlings_gdf, buffer_polygon)

    # ✅ Segment area in hectares
    segment_area_ha = combined_polygon.area / 10000  # Convert m² to hectares

    # ✅ NEW: **Faster way to get seedlings within this segment using plot_id**
    segment_plot_ids = segment_group["plot_id"].unique()
    seedlings_in_segment = seedlings_gdf[seedlings_gdf["plot_id"].isin(segment_plot_ids)]

    # ✅ Separate seedlings into categories
    inner_big_seedlings = seedlings_in_segment[
        (seedlings_in_segment["locationBB"] == "inner") & (seedlings_in_segment["class"] != 0)]
    inner_small_seedlings = seedlings_in_segment[
        (seedlings_in_segment["locationBB"] == "inner") & (seedlings_in_segment["class"] == 0)]

    outer_big_seedlings = seedlings_in_segment[
        (seedlings_in_segment["locationBB"] == "outer") & (seedlings_in_segment["class"] != 0)]
    outer_small_seedlings = seedlings_in_segment[
        (seedlings_in_segment["locationBB"] == "outer") & (seedlings_in_segment["class"] == 0)]

    # ✅ Compute total seedlings
    total_big_seedlings = len(inner_big_seedlings) + len(outer_big_seedlings)
    total_small_seedlings = len(inner_small_seedlings) + len(outer_small_seedlings)

    # ✅ Calculate density metrics
    segment_density_metrics = {
        "segment_inner_big_count": len(inner_big_seedlings),
        "segment_inner_small_count": len(inner_small_seedlings),
        "segment_outer_big_count": len(outer_big_seedlings),
        "segment_outer_small_count": len(outer_small_seedlings),
        "segment_total_big_count": total_big_seedlings,
        "segment_total_small_count": total_small_seedlings,
        "segment_density_big_total": total_big_seedlings / segment_area_ha if segment_area_ha > 0 else 0,
        "segment_density_small_total": total_small_seedlings / segment_area_ha if segment_area_ha > 0 else 0,
        "segment_density_inner_big": len(inner_big_seedlings) / segment_area_ha if segment_area_ha > 0 else 0,
        "segment_density_inner_small": len(inner_small_seedlings) / segment_area_ha if segment_area_ha > 0 else 0,
        "segment_density_outer_big": len(outer_big_seedlings) / segment_area_ha if segment_area_ha > 0 else 0,
        "segment_density_outer_small": len(outer_small_seedlings) / segment_area_ha if segment_area_ha > 0 else 0,
    }

    # print('\n******', segment_density_metrics)

    # Adjacency density metrics (in seedlings per hectare within buffer)
    buffer_area_ha = buffer_polygon.area / 10000  # Convert m² to hectares

    adjacency_density_metrics = {
        "adjacency_area_ha": buffer_area_ha if buffer_area_ha > 0 else 0,
        "adjacency_density_total": total_seedlings / buffer_area_ha if buffer_area_ha > 0 else 0,
        **{f"forest_{cls}_density": adjacency_class_counts.get(cls, 0) / buffer_area_ha if buffer_area_ha > 0 else 0 for cls in range(4)},
    }

    if not wide:
        total_plots = segment_group["plot_id"].nunique()
        seedlings_in_segment = seedlings_gdf[seedlings_gdf["plot_id"].isin(segment_group["plot_id"])]
        big_seedlings_in_segment = seedlings_in_segment[seedlings_in_segment["class"] != 0]
        # print('Overall n of seedlings: ', len(seedlings_in_segment))

        # Determine plots with at least one seedling
        plots_with_seedlings = seedlings_in_segment[seedlings_in_segment["locationBB"] != "forest"][
            "plot_id"].unique()  ###### includes forest as well! wrong
        plots_with_big_seedlings = big_seedlings_in_segment[big_seedlings_in_segment["locationBB"] != "forest"][
            "plot_id"].unique()

        # Determine inner seedlings
        inner_seedlings = seedlings_in_segment[seedlings_in_segment["locationBB"] == "inner"]  ### correct
        plots_with_inner_seedlings = inner_seedlings["plot_id"].unique()

        inner_big_seedlings = big_seedlings_in_segment[big_seedlings_in_segment["locationBB"] == "inner"]  ### correct
        plots_with_big_inner_seedlings = inner_big_seedlings["plot_id"].unique()

        # Calculate stocking percentages
        inner_stocking = len(plots_with_inner_seedlings)
        inner_big_stocking = len(plots_with_big_inner_seedlings)
        # print('N of seedlings with unique plot: ', total_stocking)
        # print('N of Inner seedlings: ', len(inner_seedlings))
        # print('N of plots with inner seedlings: ', inner_stocking)

        inner_stocking = (inner_stocking / total_plots) * 100 if total_plots > 0 else 0
        inner_big_stocking = (inner_big_stocking / total_plots) * 100 if total_plots > 0 else 0

        # print(
        #     f"Plots with Seedlings: {total_plots}, Inner Stocking: {inner_stocking:.1f}%, Total Stocking: {total_stocking:.1f}%")


    else: # Wide lines (area > 150m²)
        total_plots = segment_group["plot_id"].nunique()
        seedlings_in_segment = seedlings_gdf[seedlings_gdf["plot_id"].isin(segment_group["plot_id"])]
        inner_seedlings = seedlings_in_segment[seedlings_in_segment["locationBB"] == "inner"]
        big_inner_seedlings = inner_seedlings[inner_seedlings["class"] != 0]

        # Group by plot_id and count the number of seedlings in each plot
        big_seedling_counts = big_inner_seedlings.groupby("plot_id").size()
        inner_big_stocking = calculate_stocking(big_seedling_counts)

        # Handle inner seedlings
        inner_seedling_counts = inner_seedlings.groupby("plot_id").size()
        inner_stocking = calculate_stocking(inner_seedling_counts)

        # Convert to percentages
        inner_stocking = (inner_stocking / total_plots) * 100 if total_plots > 0 else 0
        inner_big_stocking = (inner_big_stocking / total_plots) * 100 if total_plots > 0 else 0

        # print(
        #     f"Plots with Seedlings: {total_plots}, Inner Stocking: {inner_stocking:.1f}%, Total Stocking: {total_stocking:.1f}%")

    # Side coverage metrics
    side_coverage = calculate_side_coverage(segment_group, seedlings_in_segment[seedlings_in_segment["locationBB"] != "forest"], wide)
    if not isinstance(side_coverage, dict):
        print(f"Warning: side_coverage is not a dictionary. Received: {side_coverage}")
        side_coverage = {"side0_coverage": 0.0, "side1_coverage": 0.0}

    # Collect results for this segment
    segment_metrics = {
        "SegmentID": segment_id,
        "UniqueID": unique_id,
        "segment_area": combined_polygon.area,
        "segment_area_ha": segment_area_ha,
        **segment_density_metrics,
        "inner_stocking": inner_stocking,
        "inner_big_stocking": inner_big_stocking,
        **side_coverage,
        **adjacency_density_metrics,
    }

    # Per-plot metrics
    for _, plot in segment_group.iterrows():
        plot_id = plot["plot_id"]
        plot_polygon = plot.geometry
        seedlings_in_plot = seedlings_gdf[seedlings_gdf["plot_id"] == plot_id]
        inner_seedlings_in_plot = seedlings_in_plot[seedlings_in_plot["locationBB"] == "inner"]
        big_inner_seedlings = inner_seedlings_in_plot[inner_seedlings_in_plot["class"] != 0]

        total_count = inner_seedlings_in_plot.shape[0]
        total_big_count = big_inner_seedlings.shape[0]

        plot_class_counts = inner_seedlings_in_plot["class"].value_counts().to_dict()

        results.append({
            "SegmentID": segment_id,
            "UniqueID": unique_id,
            "plot_id": plot_id,
            "geometry": plot_polygon,
            "plot_area": plot_polygon.area,
            "plot_area_ha": plot_polygon.area / 10000,  # Convert to hectares
            "plot_total_inner_count": total_count,
            'plot_big_inner_count': total_big_count,
            **{f"plot_class_{cls}_inner_count": plot_class_counts.get(cls, 0) for cls in range(4)},
            **segment_metrics,
        })

    # ✅ Free memory after processing segment
    del seedlings_in_segment, inner_big_seedlings, inner_small_seedlings
    del outer_big_seedlings, outer_small_seedlings, segment_density_metrics
    del adjacency_density_metrics, side_coverage

    return results

def process_segment(args):
    """
    Wrapper function to process a single segment in parallel.
    """
    unique_id, segment_id, segment_group, polygon_gdf, seedlings_gdf, adjacency_buffer, adjacency_gap_buffer = args
    # print(f"Processing Segment: {segment_id} | Process ID: {os.getpid()}")
    return calculate_metrics_for_segments(unique_id, segment_id, segment_group, polygon_gdf, seedlings_gdf, adjacency_buffer, adjacency_gap_buffer)

import geopandas as gpd

def assign_class_based_on_diameter(seedlings_gdf):
    """
    Assign 'class' column to 0 for seedlings with a diameter < 50 cm.
    Assumes seedlings are stored as circular buffers and calculates diameter.
    """

    # Compute diameter from geometry (assuming circular buffers)
    seedlings_gdf["diameter_cm"] = seedlings_gdf.geometry.bounds.apply(
        lambda row: max(row["maxx"] - row["minx"], row["maxy"] - row["miny"]), axis=1
    ) * 100  # Convert from meters to cm if needed

    # Assign class 0 to seedlings with diameter < 50 cm
    seedlings_gdf.loc[seedlings_gdf["diameter_cm"] < 50, "class"] = 0

    return seedlings_gdf


def process_unified_dataset(footprint_path, seedlings_path, adjacency_buffer, adjacency_gap_buffer, output_path, num_cores=16):
    """Process dataset using plot_id-based batch processing for UniqueID groups."""
    print(f"Processing dataset using batch processing (UniqueID-based)...")

    # Load datasets
    footprint_gdf = gpd.read_file(footprint_path)
    seedlings_gdf = gpd.read_file(seedlings_path)
    seedlings_gdf = assign_class_based_on_diameter(seedlings_gdf)

    # if PD == 5:
    #     seedlings_gdf["class"] = seedlings_gdf["class"] +1

    # ✅ Group by UniqueID instead of SegmentID for batch processing
    unique_id_groups = [
        (unique_id, group, footprint_gdf, seedlings_gdf, adjacency_buffer, adjacency_gap_buffer)
        for unique_id, group in footprint_gdf.groupby("UniqueID")
    ]

    with Pool(num_cores) as pool:
        with tqdm(total=len(unique_id_groups), desc="Processing UniqueID Batches") as pbar:
            results_list = []
            for result in pool.imap_unordered(process_uniqueid_batch, unique_id_groups):
                results_list.append(result)
                pbar.update()

    # ✅ Convert results to GeoDataFrame
    results_gdf = gpd.GeoDataFrame(
        [item for sublist in results_list for item in sublist],
        crs=footprint_gdf.crs
    )

    # ✅ Save results
    results_gdf.to_file(output_path, driver="GPKG")
    print(f"Processing complete. Saved to {output_path}\n")

    # ✅ Free memory after processing
    del footprint_gdf, seedlings_gdf, unique_id_groups, results_list, results_gdf

def process_uniqueid_batch(args):
    """Processes all segments for a given UniqueID in one batch."""
    unique_id, uniqueid_group, polygon_gdf, seedlings_gdf, adjacency_buffer, adjacency_gap_buffer = args
    results = []

    # print(f"Processing UniqueID: {unique_id} | Process ID: {os.getpid()}")

    for segment_id, segment_group in uniqueid_group.groupby("SegmentID"):
        segment_results = calculate_metrics_for_segments(
            unique_id, segment_id, segment_group, polygon_gdf, seedlings_gdf, adjacency_buffer, adjacency_gap_buffer
        )
        results.extend(segment_results)  # ✅ Append all segment results in the batch

    # ✅ Free memory after processing this UniqueID
    del uniqueid_group
    return results


### ----------------------------------------------

# ✅ Detect OS
IS_LINUX = platform.system() == "Linux"
# IS_LINUX = False
print('IS_LINUX', IS_LINUX)

# ✅ Set paths based on OS
if IS_LINUX:
    CONFIG_PATH = "/home/irina/RecoveryAss/Scripts/footprint/BRFN/BRFN_config.yaml"
    BASE_FOLDER = "/media/irina/data/Blueberry/DATA/"
else:  # Windows
    CONFIG_PATH = r"C:\Users\X\Documents\FalconAndSwift\BRFN\recovery_assessment\footprint\BRFN\BRFN_config.yaml"
    BASE_FOLDER = r"D:\Blueberry_NEW\7_products\vector_data\seedlings"

# ✅ Specify dataset (change to "PD25" or "PD5" as needed)
POINT_DENSITY = "PD25"

# ✅ Specify confidence of seedling detection
if POINT_DENSITY == 'PD25':
    confidence = 40
elif POINT_DENSITY == 'PD300':
    confidence = 25

# # ✅ Choose specific sites to process (or leave empty to process all)
# SELECTED_SITES = ["AtticRoad"]
SELECTED_SITES = False

def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_full_path(base_folder, relative_path, pd_version, confidence, IS_LINUX=True):
    """
    Construct a full path using base_folder and inserting PD version dynamically.
    Ensures paths are correctly formatted for both Windows and Linux.
    """

    # Format the path with dataset version (PD) and confidence value
    formatted_path = relative_path.format(PD=pd_version, conf=int(confidence))
    full_path = os.path.join(base_folder, formatted_path)
    normalized_path = os.path.normpath(full_path)
    print(f"Formatted path: {normalized_path}")
    return normalized_path


def replace_subfolder_with_metrics(file_path):
    """
    Replace the 'seedlings' subfolder with 'metrics' in the given file path.
    Ensures that the new path is correctly formatted.
    """
    path_parts = file_path.split(os.sep)  # Split path into components
    if "seedlings" in path_parts:
        path_parts[path_parts.index("seedlings")] = "metrics"  # Replace 'seedlings' with 'metrics'

    return os.sep.join(path_parts)  # Reconstruct the path with correct separators

def process_site(site_name, site_config, base_folder, point_density, confidence, IS_LINUX):
    """Process a single site by constructing paths dynamically."""
    chm_path = get_full_path(base_folder, site_config["chm"], point_density, confidence, IS_LINUX)
    seedlings_path = get_full_path(base_folder, site_config["seedlings"], point_density, confidence, IS_LINUX)
    footprint_path = get_full_path(base_folder, site_config["ground_footprint"], point_density, confidence, IS_LINUX)

    output_dir = os.path.dirname(replace_subfolder_with_metrics(seedlings_path))
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{site_name}_metrics_{point_density}.gpkg")

    print(f"Processing {site_name} for {point_density}...")
    print(f"  - Footprint: {footprint_path}")
    # print(f"  - CHM: {chm_path}")
    print(f"  - Seedlings: {seedlings_path}")
    print(f"  - Output: {output_path}")

    process_unified_dataset(footprint_path, seedlings_path,
                            adjacency_buffer = 20, adjacency_gap_buffer = 2, output_path = output_path, num_cores = 1)
    print(f"{site_name} processing complete!\n")

def main():
    """Process specified sites for the given dataset."""
    print(f"  - Config File: {CONFIG_PATH}")
    print(f"  - Base Folder: {BASE_FOLDER}")

    config = load_config(CONFIG_PATH)

    # ✅ Ensure correct base folder is used
    base_folder = BASE_FOLDER

    # ✅ Filter selected sites
    all_sites = config["sites_for_recovery_assessment"]
    sites = {k: v for k, v in all_sites.items() if not SELECTED_SITES or k in SELECTED_SITES}

    print('Confidence: ', confidence)

    for site_name, site_config in sites.items():
        process_site(site_name, site_config, base_folder, POINT_DENSITY, confidence, IS_LINUX)

    print("All processing complete!")

if __name__ == "__main__":
    main()
