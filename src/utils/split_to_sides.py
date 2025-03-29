import os
import math
import logging
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiLineString, GeometryCollection
from shapely.ops import split, linemerge
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf


# -----------------------------
# Geometry helper functions
# -----------------------------
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
    """
    if polygon.is_empty or not polygon.is_valid:
        return set()
    edge_coords = polygon.exterior.coords
    edge_points = {(round(coord[0], precision), round(coord[1], precision)) for coord in edge_coords}
    return edge_points


def sort_segments_and_find_pairs(gdf):
    """
    Sort segments by orientation, assign sides, and find pairs based on shared edge points.
    Returns a GeoDataFrame with a new "plot_id" field.
    """

    def process_unique_id(subset):
        orientation = determine_orientation(subset.geometry.iloc[0])
        subset["centroid_x"] = subset.geometry.centroid.x
        subset["centroid_y"] = subset.geometry.centroid.y
        subset["edge_points"] = subset.geometry.apply(get_edge_points)
        # If PartID is not present, use the index.
        if "PartID" not in subset.columns:
            subset = subset.reset_index().rename(columns={"index": "PartID"})
        else:
            subset = subset.sort_values("PartID")
        subset["side"] = 1  # Default side 1.
        half_rows = len(subset) // 2
        subset.iloc[half_rows:, subset.columns.get_loc("side")] = 0  # Second half: side 0.
        subset["plot_id"] = -1
        segment_id = 0
        side_0 = subset[subset["side"] == 0]
        side_1 = subset[subset["side"] == 1]
        for idx_0, row_0 in side_0.iterrows():
            if row_0.geometry.area < 7:
                continue
            for idx_1, row_1 in side_1.iterrows():
                if row_1.geometry.area < 7:
                    continue
                shared_points = row_0["edge_points"].intersection(row_1["edge_points"])
                if len(shared_points) >= 2:  # Found a pair.
                    subset.at[idx_0, "plot_id"] = segment_id
                    subset.at[idx_1, "plot_id"] = segment_id
                    segment_id += 1
        return subset

    return gdf.groupby("UniqueID", group_keys=False).apply(process_unique_id)


def update_path_with_id(input_path, output_dir):
    """
    Update the input path to include '_ID' and return a new path in the output directory.
    """
    filename = os.path.basename(input_path)
    if filename.endswith(".shp"):
        updated_filename = filename.replace(".shp", "_ID.gpkg")
    elif filename.endswith(".gpkg"):
        updated_filename = filename.replace(".gpkg", "_ID.gpkg")
    else:
        raise ValueError("Unsupported file format. Only '.shp' and '.gpkg' are supported.")
    return os.path.join(output_dir, updated_filename)


# ---------------------------
# Main processing function using Hydra
# ---------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up logging (default INFO level if not specified)
    log_level = cfg.get("logging", {"level": "INFO"}).get("level", "INFO")
    logging.basicConfig(level=log_level)
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Use the UniqueID output from the previous split_to_plots step.
    # Output directory will be the same as the input file directory.
    output_dir = os.path.dirname(cfg.dataset.ground_footprint)
    footprint_path = update_path_with_id(cfg.dataset.ground_footprint, output_dir)

    # Build the input file path for segments.
    # Assume the split_to_plots step produced a file with name ending in
    # "_ID_segments{segment_area}m2.gpkg" and with prefix "11_"
    segment_area = int(cfg.split_to_side.segment_area)
    input_filename = os.path.basename(footprint_path).replace("_ID.gpkg", f"_footprint_ID_segments{segment_area}m2.gpkg")
    input_path = os.path.join(output_dir, input_filename)
    print('Working with: ', input_filename)

    # Build the output path by replacing "11_" with "12_" and suffix with "_sides.gpkg"
    output_filename = input_filename.replace(f"footprint_ID_segments{segment_area}m2.gpkg", "sides.gpkg")
    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Input for split_to_side: {input_path}")
    logging.info(f"Output for split_to_side: {output_path}")
    logging.info(f"Segment area: {segment_area} m2")

    # Read input GeoDataFrame
    gdf = gpd.read_file(input_path)
    logging.info(f"Read {len(gdf)} segments from {input_path}")

    # Process segments to sort and find pairs.
    paired_gdf = sort_segments_and_find_pairs(gdf)
    logging.info("Finished processing segments for pairing.")

    # Optionally filter small polygons using min_area from configuration.
    min_area = cfg.split_to_side.get("min_area", 5)
    paired_gdf = paired_gdf[paired_gdf.geometry.area >= min_area]
    logging.info(f"After filtering, {len(paired_gdf)} segments remain.")

    # Save the paired segments GeoDataFrame
    paired_gdf.to_file(output_path, driver="GPKG")
    print(f"Paired segments saved to: {output_path}")


if __name__ == "__main__":
    main()
