import os
import geopandas as gpd
import pandas as pd
from rasterio.features import geometry_mask
from tqdm import tqdm
import rasterio
from rasterio.mask import mask
import numpy as np
from multiprocessing import Pool, cpu_count
import yaml
import glob

# parallelization works for this script
# also using spatial indexing

def debug_adjacency(seedlings_gdf, seedling_sindex, buffer_polygon):
    """
    Efficient adjacency count using spatial indexing.
    """
    # Get bounding box matches first
    possible_matches = list(seedling_sindex.intersection(buffer_polygon.bounds))
    adjacency_seedlings = seedlings_gdf.iloc[possible_matches]

    # Apply exact filtering only to matches
    adjacency_seedlings = adjacency_seedlings[adjacency_seedlings.geometry.within(buffer_polygon)]

    total_seedlings = len(adjacency_seedlings)
    adjacency_class_counts = adjacency_seedlings["class"].value_counts().to_dict()

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


def calculate_metrics_for_segments(unique_id, segment_id, segment_group, polygon_gdf,polygon_sindex,
                                   seedlings_gdf, seedling_sindex,
                                   adjacency_buffer, adjacency_gap_buffer):
    """
    Calculate metrics for each segment in the unified dataset.
    """
    results = []

    wide = False  # Initialize wide as False at the start of each segment
    combined_polygon = segment_group.geometry.unary_union
    if combined_polygon.area > 150:
        wide = True
        #print('WIDE LINE')

    # create adjacency area
    buffer_polygon = combined_polygon.buffer(adjacency_buffer).difference(combined_polygon.buffer(adjacency_gap_buffer))

    # also difference out nearby segments.
    nearby_polygons = polygon_gdf[polygon_gdf.intersects(buffer_polygon)]
    buffer_polygon = buffer_polygon.difference(nearby_polygons.buffer(adjacency_gap_buffer).unary_union)

    # Detect  adjacency counts
    total_seedlings, adjacency_class_counts = debug_adjacency(seedlings_gdf, seedling_sindex, buffer_polygon)

    # Total density metrics for the segment (in seedlings per hectare)
    segment_area_ha = combined_polygon.area / 10000  # Convert m² to hectares

    possible_seedlings = list(seedling_sindex.intersection(combined_polygon.bounds))
    seedlings_in_segment = seedlings_gdf.iloc[possible_seedlings]
    seedlings_in_segment = seedlings_in_segment[seedlings_in_segment.geometry.within(combined_polygon)]

    # Density metrics
    inner_seedlings = seedlings_in_segment[seedlings_in_segment["locationBB"] == "inner"]
    outer_seedlings = seedlings_in_segment[seedlings_in_segment["locationBB"] == "outer"]

    segment_density_metrics = {
        "segment_inner_count": len(inner_seedlings),
        "segment_outer_count": len(outer_seedlings),
        "segment_density_total": len(seedlings_in_segment) / segment_area_ha if segment_area_ha > 0 else 0,
        "segment_density_inner": len(inner_seedlings) / segment_area_ha if segment_area_ha > 0 else 0,
        "segment_density_outer": len(outer_seedlings) / segment_area_ha if segment_area_ha > 0 else 0,
    }

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
        print('Overall n of seedlings: ', len(seedlings_in_segment))

        # Determine plots with at least one seedling
        plots_with_seedlings = seedlings_in_segment[seedlings_in_segment["locationBB"] != "forest"][
            "plot_id"].unique()  ###### includes forest as well! wrong

        # Determine inner seedlings
        inner_seedlings = seedlings_in_segment[seedlings_in_segment["locationBB"] == "inner"]  ### correct
        plots_with_inner_seedlings = inner_seedlings["plot_id"].unique()

        # Calculate stocking percentages
        inner_stocking = len(plots_with_inner_seedlings)
        total_stocking = len(plots_with_seedlings)
        # print('N of seedlings with unique plot: ', total_stocking)
        # print('N of Inner seedlings: ', len(inner_seedlings))
        # print('N of plots with inner seedlings: ', inner_stocking)

        inner_stocking = (inner_stocking / total_plots) * 100 if total_plots > 0 else 0
        total_stocking = (total_stocking / total_plots) * 100 if total_plots > 0 else 0

        # print(
        #     f"Plots with Seedlings: {total_plots}, Inner Stocking: {inner_stocking:.1f}%, Total Stocking: {total_stocking:.1f}%")

    else: # Wide lines (area > 150m²)
        total_plots = segment_group["plot_id"].nunique()
        seedlings_in_segment = seedlings_gdf[seedlings_gdf["plot_id"].isin(segment_group["plot_id"])]

        # Group by plot_id and count the number of seedlings in each plot
        seedling_counts = seedlings_in_segment.groupby("plot_id").size()

        # Count plots based on the number of seedlings (1+ counts as 0.5, 2+ counts as 1)
        total_stocking = 0

        for count in seedling_counts:

            if count >= 2:

                total_stocking += 1  # Count as 1 for 2+ seedlings

            elif count == 1:

                total_stocking += 0.5  # Count as 0.5 for 1+ seedlings

        # Handle inner seedlings

        inner_seedlings = seedlings_in_segment[seedlings_in_segment["locationBB"] == "inner"]

        inner_seedling_counts = inner_seedlings.groupby("plot_id").size()

        inner_stocking = 0

        for count in inner_seedling_counts:

            if count >= 2:

                inner_stocking += 1  # Count as 1 for 2+ seedlings

            elif count == 1:

                inner_stocking += 0.5  # Count as 0.5 for 1+ seedlings

        # Convert to percentages
        inner_stocking = (inner_stocking / total_plots) * 100 if total_plots > 0 else 0
        total_stocking = (total_stocking / total_plots) * 100 if total_plots > 0 else 0

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
        "total_stocking": total_stocking,
        **side_coverage,
        **adjacency_density_metrics,
    }

    # Per-plot metrics
    for _, plot in segment_group.iterrows():
        plot_id = plot["plot_id"]
        plot_polygon = plot.geometry
        seedlings_in_plot = seedlings_gdf[seedlings_gdf["plot_id"] == plot_id]
        inner_count = seedlings_in_plot[seedlings_in_plot["locationBB"] == "inner"].shape[0]
        outer_count = seedlings_in_plot[seedlings_in_plot["locationBB"] == "outer"].shape[0]
        total_count = inner_count + outer_count
        plot_class_counts = seedlings_in_plot["class"].value_counts().to_dict()

        results.append({
            "SegmentID": segment_id,
            "UniqueID": unique_id,
            "plot_id": plot_id,
            "geometry": plot_polygon,
            "plot_area": plot_polygon.area,
            "plot_area_ha": plot_polygon.area / 10000,  # Convert to hectares
            "plot_inner_count": inner_count,
            "plot_outer_count": outer_count,
            "plot_total_count": total_count,
            **{f"plot_class_{cls}_count": plot_class_counts.get(cls, 0) for cls in range(4)},
            **segment_metrics,
        })

    return results

def process_segment(args):
    """
    Wrapper function to process a single segment in parallel.
    """
    unique_id, segment_id, segment_group, polygon_gdf, polygon_sindex, seedlings_gdf, seedling_sindex, adjacency_buffer, adjacency_gap_buffer = args
    return calculate_metrics_for_segments(unique_id, segment_id, segment_group, polygon_gdf, polygon_sindex, seedlings_gdf, seedling_sindex, adjacency_buffer, adjacency_gap_buffer)

def process_unified_dataset(polygon_path, seedlings_path,
                            adjacency_buffer, adjacency_gap_buffer, output_path, num_cores):
    polygons_gdf = gpd.read_file(polygon_path)
    seedlings_gdf = gpd.read_file(seedlings_path)

    if seedlings_gdf.crs != polygons_gdf.crs:
        seedlings_gdf = seedlings_gdf.to_crs(polygons_gdf.crs)

    # build spatial index
    polygon_sindex = polygons_gdf.sindex
    seedling_sindex = seedlings_gdf.sindex

    segment_groups = [
        (unique_id, segment_id, seg_group, polygons_gdf, polygon_sindex, seedlings_gdf, seedling_sindex,
         adjacency_buffer, adjacency_gap_buffer)
        for (unique_id, segment_id), seg_group in polygons_gdf.groupby(["UniqueID", "SegmentID"])
    ]

    with Pool(num_cores) as pool:
        with tqdm(total=len(segment_groups), desc="Processing Segments") as pbar:
            results_list = []
            for result in pool.imap_unordered(process_segment, segment_groups):
                results_list.append(result)
                pbar.update()

    results_gdf = gpd.GeoDataFrame([item for sublist in results_list for item in sublist], crs=polygons_gdf.crs)
    results_gdf.to_file(output_path, driver="GPKG")

def process_site_config(config_path):
    """
    Process a single site configuration file.
    If you want to run a single site, scroll to the bottom.

    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    sitename = config['parameters']['sitename']
    adjacency_buffer = 20
    adjacency_gap_buffer = 2
    num_cores = 1

    # Always use ground_footprint from PD300
    if "datasets_PD300" not in config:
        raise ValueError(f"Missing 'datasets_PD300' in {config_path}. Ground footprint is required.")

    polygon_path = config["datasets_PD300"]["ground_footprint"]

    dataset_keys = ["datasets_PD300", "datasets_PD25", "datasets_PD5"]

    for dataset_key in dataset_keys:
        if dataset_key in config:  # Ensure the dataset exists before processing
            dataset = config[dataset_key]

            seedlings_path = dataset["seedlings"]
            output_dir = dataset["assess_output_dir"]

            # Define output filename
            output_path = os.path.join(output_dir, f"{sitename}_metrics1_{dataset_key}.gpkg")

            print(f"Processing {dataset_key} for site: {sitename}...")
            process_unified_dataset(polygon_path, seedlings_path,
                                    adjacency_buffer, adjacency_gap_buffer, output_path, num_cores)
            print(f"{dataset_key} processing complete!\n")


def main():
    """Iterate through all config files in 'config_files_by_site' and process each."""
    config_dir = r"C:\Users\X\Documents\FalconAndSwift\BRFN\recovery_assessment\footprint\BRFN\config_files_by_site"

    # Get all YAML files in the directory
    config_files = glob.glob(os.path.join(config_dir, "*_config.yaml"))

    if not config_files:
        print("No configuration files found.")
        return

    for config_path in config_files:
        print(f"\nProcessing config file: {config_path}")
        process_site_config(config_path)

    print("All processing complete!")


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     # File paths
#     polygon_path = r"D:\Blueberry_NEW\7_products\test2\test_plots.gpkg"
#     seedlings_path = r"D:\Blueberry_NEW\7_products\test2\seedlings\test_seedlings.gpkg"
#     raster_path = r"D:\Blueberry_NEW\7_products\test2\trails\vrt_testing.vrt"
#     output_path = r"D:\Blueberry_NEW\7_products\test2\assessment\test_metrics1_new12.gpkg"
#     raster_threshold = 30
#     adjacency_buffer = 20 # buffer into adjacency
#     adjacency_gap_buffer = 2 # buffer around segment of interest to remove
#     num_cores = 8
#
#     # Process the unified dataset
#     process_unified_dataset(polygon_path, seedlings_path, raster_path,
#                             raster_threshold, adjacency_buffer,adjacency_gap_buffer, output_path, num_cores)
#
#     print("Processing complete!")
