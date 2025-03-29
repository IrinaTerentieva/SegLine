import os
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
from tqdm import tqdm
from rasterio.features import geometry_mask
import yaml
import glob

def read_clipped_data(src, geom):
    """
    Reads and clips raster data to the exact geometry.
    """
    geom_mapping = [geom.__geo_interface__]

    try:
        clipped_data, clipped_transform = mask(src, geom_mapping, crop=True, nodata=np.nan)

        # Create a binary mask for the geometry
        binary_mask = geometry_mask(
            [geom_mapping[0]], out_shape=clipped_data.shape[1:], transform=clipped_transform, invert=True
        )

        # Apply the binary mask to keep only inner pixels
        inner_data = np.where(binary_mask, clipped_data[0], np.nan)

        return inner_data
    except rasterio.errors.RasterioIOError:
        return None

def calculate_chm_coverage(raster_data, min_height_m, max_height_m):
    """
    Calculate the percentage of pixels within a height range.
    """
    valid_pixels = ~np.isnan(raster_data)
    total_pixels = np.sum(valid_pixels)

    if total_pixels == 0:
        return 0

    in_range_pixels = ((raster_data >= min_height_m) & (raster_data <= max_height_m)).sum()
    return int((in_range_pixels / total_pixels) * 100)

def calculate_trails_coverage(raster_data, threshold):
    """
    Calculate the percentage of pixels above a threshold
    """
    valid_pixels = ~np.isnan(raster_data)
    total_pixels = np.sum(valid_pixels)

    if total_pixels == 0:
        return 0

    in_range_pixels = (raster_data >= threshold).sum()
    return int((in_range_pixels / total_pixels) * 100)


def process_segment(polygon_gdf, chm_maskedseedlings_path, trails_path, trails_threshold, adjacency_buffer,
                    adjacency_gap_buffer):
    """
    Process a single segment to get shrub, tree, and trail coverages.
    """

    results = []

    # Open both rasters
    with rasterio.open(chm_maskedseedlings_path) as src_chm, rasterio.open(trails_path) as src_trails:
        for unique_id, unique_group in tqdm(polygon_gdf.groupby("UniqueID"), desc="Processing UniqueIDs"):
            for segment_id, segment_group in unique_group.groupby("SegmentID"):
                combined_polygon = segment_group.geometry.unary_union

                if combined_polygon.is_empty:
                    continue

                # Compute adjacency buffer once
                buffer_polygon = combined_polygon.buffer(adjacency_buffer).difference(
                    combined_polygon.buffer(adjacency_gap_buffer)
                )

                # Exclude nearby segments
                nearby_polygons = polygon_gdf[polygon_gdf.intersects(buffer_polygon)]
                buffer_polygon = buffer_polygon.difference(nearby_polygons.buffer(adjacency_gap_buffer).unary_union)

                # Read CHM raster data for segment and adjacency
                segment_raster_data = read_clipped_data(src_chm, combined_polygon)
                buffer_raster_data = read_clipped_data(src_chm, buffer_polygon)

                if segment_raster_data is None:
                    continue

                # Compute segment shrub/tree coverage
                shrub_30_60cm_coverage = calculate_chm_coverage(segment_raster_data, 0.3, 0.6)
                shrub_60_200cm_coverage = calculate_chm_coverage(segment_raster_data, 0.6, 2.0)
                shrub_200_300cm_coverage = calculate_chm_coverage(segment_raster_data, 2.0, 3.0)
                total_shrub_coverage = shrub_30_60cm_coverage + shrub_60_200cm_coverage + shrub_200_300cm_coverage
                tree_coverage = calculate_chm_coverage(segment_raster_data, 3.0, 60.0)
                seedling_coverage = calculate_chm_coverage(segment_raster_data, -1.0, -1.0)
                tree_coverage = tree_coverage + seedling_coverage

                # Compute adjacency coverage
                adjacency_shrub_30_60cm_coverage = calculate_chm_coverage(buffer_raster_data, 0.3, 0.6)
                adjacency_shrub_60_200cm_coverage = calculate_chm_coverage(buffer_raster_data, 0.6, 2.0)
                adjacency_shrub_200_300cm_coverage = calculate_chm_coverage(buffer_raster_data, 2.0, 3.0)
                adjacency_total_shrub_coverage = adjacency_shrub_30_60cm_coverage + adjacency_shrub_60_200cm_coverage + adjacency_shrub_200_300cm_coverage

                adjacency_tree_coverage = calculate_chm_coverage(buffer_raster_data, 3.0, 60.0)
                adjacency_seedling_coverage = calculate_chm_coverage(buffer_raster_data, -1.0, -1.0)
                adjacency_tree_coverage = adjacency_tree_coverage + adjacency_seedling_coverage

                # Compute trail coverage using the trails raster
                segment_trail_coverage = calculate_trails_coverage(read_clipped_data(src_trails, combined_polygon),
                                                                   trails_threshold)
                adjacency_trail_coverage = calculate_trails_coverage(read_clipped_data(src_trails, buffer_polygon),
                                                                     trails_threshold)

                segment_metrics = {
                    "SegmentID": segment_id,
                    "UniqueID": unique_id,
                    "segment_area": combined_polygon.area,
                    "segment_area_ha": combined_polygon.area / 10000,
                    'segment_30_60cm_shrub_coverage': shrub_30_60cm_coverage,
                    'segment_60_200cm_shrub_coverage': shrub_60_200cm_coverage,
                    'segment_200_300cm_shrub_coverage': shrub_200_300cm_coverage,
                    'segment_total_shrub_coverage': total_shrub_coverage,
                    'segment_tree_coverage': tree_coverage,
                    'adjacency_30_60cm_shrub_coverage': adjacency_shrub_30_60cm_coverage,
                    'adjacency_60_200cm_shrub_coverage': adjacency_shrub_60_200cm_coverage,
                    'adjacency_200_300cm_shrub_coverage': adjacency_shrub_200_300cm_coverage,
                    'adjacency_total_shrub_coverage': adjacency_total_shrub_coverage,
                    'adjacency_tree_coverage': adjacency_tree_coverage,
                    'segment_trail_coverage': segment_trail_coverage,
                    'adjacency_trail_coverage': adjacency_trail_coverage,
                }

                for _, plot in segment_group.iterrows():
                    plot_id = plot["plot_id"]
                    plot_polygon = plot.geometry

                    # Read plot-level raster data
                    plot_raster_data = read_clipped_data(src_chm, plot_polygon)
                    plot_trail_data = read_clipped_data(src_trails, plot_polygon)

                    if plot_raster_data is None or plot_trail_data is None:
                        continue

                    # Calculate shrub coverages for the plot
                    plot_shrub_30_60cm_coverage = calculate_chm_coverage(plot_raster_data, 0.3, 0.6)
                    plot_shrub_60_200cm_coverage = calculate_chm_coverage(plot_raster_data, 0.6, 2.0)
                    plot_shrub_200_300cm_coverage = calculate_chm_coverage(plot_raster_data, 2.0, 3.0)
                    plot_total_shrub_coverage = plot_shrub_30_60cm_coverage + plot_shrub_60_200cm_coverage + plot_shrub_200_300cm_coverage

                    plot_tree_coverage = calculate_chm_coverage(plot_raster_data, 3.0, 60.0)
                    plot_seedling_coverage = calculate_chm_coverage(plot_raster_data, -1.0, -1.0)
                    plot_tree_coverage = plot_tree_coverage + plot_seedling_coverage

                    # Calculate trail coverage for the plot
                    plot_trail_coverage = calculate_trails_coverage(plot_trail_data, trails_threshold)

                    results.append({
                        "SegmentID": segment_id,
                        "UniqueID": unique_id,
                        "plot_id": plot_id,
                        "geometry": plot_polygon,
                        "plot_area": plot_polygon.area,
                        "plot_area_ha": plot_polygon.area / 10000,
                        'plot_30_60cm_shrub_coverage': plot_shrub_30_60cm_coverage,
                        'plot_60_200cm_shrub_coverage': plot_shrub_60_200cm_coverage,
                        'plot_200_300cm_shrub_coverage': plot_shrub_200_300cm_coverage,
                        'plot_total_shrub_coverage': plot_total_shrub_coverage,
                        'plot_tree_coverage': plot_tree_coverage,
                        'plot_trail_coverage': plot_trail_coverage,
                        **segment_metrics,
                    })

    return results


def process_unified_dataset(polygon_path, chm_maskedseedlings_path, trails_path, trails_threshold,
                            adjacency_buffer, adjacency_gap_buffer, output_path):
    """
    Process the unified dataset and calculate metrics.
    """
    polygons_gdf = gpd.read_file(polygon_path)
    polygons_gdf["geometry"] = polygons_gdf["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

    results = process_segment(polygons_gdf, chm_maskedseedlings_path, trails_path, trails_threshold, adjacency_buffer, adjacency_gap_buffer)

    results_gdf = gpd.GeoDataFrame(results, crs=polygons_gdf.crs)
    results_gdf.to_file(output_path, driver="GPKG")


# File paths
# polygon_path = r"D:\Blueberry_NEW\7_products\test2\test_plots.gpkg"
# chm_maskedseedlings_path = r"D:\Blueberry_NEW\7_products\test2\chm\test_chm_masked_seedlings.tif"
# output_path = r"D:\Blueberry_NEW\7_products\test2\assessment\test_metrics2_new3.gpkg"
# adjacency_buffer = 20
# adjacency_gap_buffer = 2

# process_unified_dataset(polygon_path, chm_maskedseedlings_path, adjacency_buffer, adjacency_gap_buffer, output_path)
# print("Processing complete!")

def process_site(config_path):
    """
    Process a single site based on its configuration file.
    scroll to the bottom if you want to process just a single file instead.
    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    sitename = config['parameters']['sitename']
    adjacency_buffer = 20
    adjacency_gap_buffer = 2
    trails_threshold = 30

    # Always use ground_footprint from datasets_PD300
    if "datasets_PD300" not in config:
        raise ValueError(f"datasets_PD300 is missing in {config_path}. Ground footprint cannot be found.")

    ground_footprint_path = config["datasets_PD300"]["ground_footprint"]

    # List of datasets to process
    dataset_keys = ["datasets_PD300", "datasets_PD25", "datasets_PD5"]

    for dataset_key in dataset_keys:
        if dataset_key in config:  # Ensure the key exists in the config file
            dataset = config[dataset_key]

            chm_maskedseedlings_path = dataset['chm_masked']
            trails_path = dataset['trails']
            output_dir = dataset['assess_output_dir']  # PD-specific output directory

            output_path = os.path.join(output_dir, f"{sitename}_metrics2_{dataset_key}.gpkg")

            print(f"Processing {dataset_key} for site: {sitename}...")
            process_unified_dataset(
                ground_footprint_path,
                chm_maskedseedlings_path,
                trails_path,
                trails_threshold,
                adjacency_buffer,
                adjacency_gap_buffer,
                output_path
            )
            print(f"{dataset_key} processing complete for site: {sitename}\n")


def main():
    config_dir = r"C:\Users\X\Documents\FalconAndSwift\BRFN\recovery_assessment\footprint\BRFN\config_files_by_site"

    # Find all YAML files in the config directory
    config_files = glob.glob(os.path.join(config_dir, "*_config.yaml"))

    print(f"Found {len(config_files)} config files. Processing all sites...\n")

    for config_path in config_files:
        print(f"Processing site from config file: {config_path}")
        process_site(config_path)

    print("All sites processing complete!")


if __name__ == "__main__":
    main()



# using config file to do this
# def main():
#     config_path = r"C:\Users\X\Documents\FalconAndSwift\BRFN\recovery_assessment\footprint\BRFN\config_files_by_site\PA2-W2(West)-SouthSikanniRoad_config.yaml"
#     with open(config_path, "r") as config_file:
#         config = yaml.safe_load(config_file)
#
#     sitename = config['parameters']['sitename']
#     adjacency_buffer = 20
#     adjacency_gap_buffer = 2
#
#     # Always use ground_footprint from datasets_PD300
#     if "datasets_PD300" not in config:
#         raise ValueError("datasets_PD300 is missing from the config file. Ground footprint cannot be found.")
#
#     ground_footprint_path = config["datasets_PD300"]["ground_footprint"]  # Always use PD300 footprint
#
#     # List of datasets to process
#     dataset_keys = ["datasets_PD300", "datasets_PD25", "datasets_PD5"]
#
#     for dataset_key in dataset_keys:
#         if dataset_key in config:  # Ensure the key exists in the config file
#             dataset = config[dataset_key]
#
#             chm_maskedseedlings_path = dataset['chm_masked']
#             output_dir = dataset['assess_output_dir']  # Now using the PD-specific output directory
#
#             output_path = os.path.join(output_dir, f"{sitename}_metrics2_{dataset_key}.gpkg")
#
#             print(f"Processing {dataset_key}...")
#             process_unified_dataset(ground_footprint_path, chm_maskedseedlings_path, adjacency_buffer,
#                                     adjacency_gap_buffer, output_path)
#             print(f"{dataset_key} processing complete!\n")
#
#     print("All processing complete!")
#
# if __name__ == "__main__":
#     main()