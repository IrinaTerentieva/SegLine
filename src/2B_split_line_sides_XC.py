import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import yaml
import os


def determine_orientation(geometry):
    """
    Determine whether a segment is east-west or north-south based on its bounding box.
    """
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    x_diff = bounds[2] - bounds[0]
    y_diff = bounds[3] - bounds[1]
    orientation = "east-west" if x_diff > y_diff else "north-south"
    return orientation


def get_edge_points(polygon, precision=3):
    """
    Extract all edge points from a polygon's exterior.

    Args:
        polygon (Polygon): The input polygon geometry.
        precision (int): Decimal precision for rounding the coordinates.

    Returns:
        set[tuple]: Set of edge points as tuples with rounded coordinates.
    """
    if polygon.is_empty or not polygon.is_valid:
        return set()

    # Extract all edge points from the polygon's exterior
    edge_coords = polygon.exterior.coords
    edge_points = {(round(coord[0], precision), round(coord[1], precision)) for coord in edge_coords}

    return edge_points


def sort_segments_and_find_pairs(gdf):
    """
    Sort segments by orientation, assign sides, and find pairs based on shared edge points.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame with geometries.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with SegmentID assigned to paired polygons.
    """

    def process_unique_id(subset):
        # Print the subset being processed
        # print(f"\nProcessing UniqueID: {subset['UniqueID'].iloc[0]}")
        # print(f"Initial subset length: {len(subset)}")

        # Determine orientation
        orientation = determine_orientation(subset.geometry.iloc[0])
        # print(f"Orientation determined: {orientation}")

        # Calculate centroids and extract edge points
        subset["centroid_x"] = subset.geometry.centroid.x
        subset["centroid_y"] = subset.geometry.centroid.y
        subset["edge_points"] = subset.geometry.apply(get_edge_points)
        # print(f"Calculated centroids and edge points for {len(subset)} rows.")

        # Determine the number of rows and split into two groups
        num_rows = len(subset)
        half_rows = num_rows // 2
        # print(f"Total rows: {num_rows}, Rows per side: {half_rows}")

        # Sort rows by their FID/index and split
        subset = subset.sort_values('PartID')
        # print("Subset sorted by index.")

        subset["side"] = 1  # Default to side 1
        subset.iloc[half_rows:, subset.columns.get_loc("side")] = 0  # Assign side 0 to the second group
        # print(f"Sides assigned:\n{subset[['side']]}")

        # Assign SegmentID starting from 0 for each pair
        subset["SegmentID"] = -1
        segment_id = 0
        side_0 = subset[subset["side"] == 0]
        side_1 = subset[subset["side"] == 1]

        # print(f"Side 0 count: {len(side_0)}, Side 1 count: {len(side_1)}")

        # Pair matching based on shared edge points
        for idx_0, row_0 in side_0.iterrows():
            if row_0.geometry.area < 7:
                continue  # Skip small areas for Side 0
            for idx_1, row_1 in side_1.iterrows():
                if row_1.geometry.area < 7:
                    continue  # Skip small areas for Side 1

                shared_points = row_0["edge_points"].intersection(row_1["edge_points"])

                if len(shared_points) >= 2:  # Found a pair
                    subset.at[idx_0, "SegmentID"] = segment_id
                    subset.at[idx_1, "SegmentID"] = segment_id
                    segment_id += 1

        return subset

    # Apply to a specific UniqueID for debugging
    # print("Starting to process GeoDataFrame...")
    # gdf = gdf[gdf["UniqueID"] == "GWdTUPLF"]  # Filter to specific UniqueID
    # print(f"Filtered GeoDataFrame length: {len(gdf)}")
    gdf = gdf.groupby("UniqueID", group_keys=False).apply(process_unique_id)
    # print("Finished processing GeoDataFrame.")

    return gdf


def update_path_with_id(input_path, output_dir):
    """
    Update the input path to include '_ID' and ensure it is saved as a GeoPackage.

    Args:
        input_path (str): Path to the input file.
        output_dir (str): Directory where the updated file will be saved.

    Returns:
        str: Updated path with '_ID.gpkg'.
    """
    filename = os.path.basename(input_path)
    if filename.endswith(".shp"):
        updated_filename = filename.replace(".shp", "_ID.gpkg")
    elif filename.endswith(".gpkg"):
        updated_filename = filename.replace(".gpkg", "_ID.gpkg")
    else:
        raise ValueError("Unsupported file format. Only '.shp' and '.gpkg' are supported.")
    return os.path.join(output_dir, updated_filename)


def filter_small_polygons(gdf, min_area=5):
    """
    Filter out polygons with an area smaller than the specified minimum.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame containing polygons.
        min_area (float): The minimum area threshold (in square meters).

    Returns:
        GeoDataFrame: A GeoDataFrame with polygons larger than or equal to the minimum area.
    """
    # Ensure geometries are valid before calculating area
    # gdf['geometry'] = gdf['geometry'].apply(fix_invalid_geometry)
    gdf = gdf[gdf['geometry'].notna()]  # Drop rows with invalid geometries

    # Filter polygons based on area
    filtered_gdf = gdf[gdf['geometry'].area >= min_area]
    return filtered_gdf

# Load configuration
config_path = "C:/Users/X/Documents/FalconAndSwift/BRFN/recovery_assessment/footprint/BRFN/test.yaml"
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

output_dir = config['datasets']['output_dir']

# Paths and parameters from config
footprint_path = update_path_with_id(config['datasets']['ground_footprint'], output_dir)
segment_area = int(config['parameters']['segment_area'])

input_path = footprint_path.replace("10_", "11_").replace("_ID.gpkg", f"_segments{segment_area}m2.gpkg")

# Construct output path, replacing the prefix 11_ with 12_
output_path = input_path.replace("11_", "12_").replace(f"_ID_segments{segment_area}m2.gpkg", f"_sides.gpkg")

# Load GeoDataFrame
gdf = gpd.read_file(input_path)

# Sort the segments and find pairs
paired_gdf = sort_segments_and_find_pairs(gdf)

# Save the paired GeoDataFrame
paired_gdf.to_file(output_path, driver="GPKG")
print(f"Paired segments saved to: {output_path}")
