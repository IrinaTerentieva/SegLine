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


def sort_segments_and_find_pairs(subset):
    """
    Sort segments by orientation, assign sides, find pairs, and assign pair_id.
    ✅ PRESERVES the original plot_id from split_to_plots.
    Returns a subset with side, pair_id, and segment_id fields.
    """
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

    # ✅ FIX: Use pair_id for pairing, preserve original plot_id
    # The plot_id from split_to_plots is already in the dataframe and should NOT be overwritten
    subset["pair_id"] = -1  # Used for pairing left/right segments
    subset["segment_id"] = -1  # Initialize segment_id

    pair_id = 0
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
                subset.at[idx_0, "pair_id"] = pair_id
                subset.at[idx_1, "pair_id"] = pair_id
                pair_id += 1

    return subset


def assign_segment_ids(gdf):
    """
    Assign segment_id within each pair_id (NOT plot_id).
    ✅ PRESERVES the original plot_id from split_to_plots.
    For paired segments (pair_id >= 0), both sides get sequential segment_ids.
    For unpaired segments (pair_id == -1), assign unique segment_ids.
    """
    # Process paired segments (pair_id >= 0)
    paired_mask = gdf['pair_id'] >= 0
    if paired_mask.any():
        paired_gdf = gdf[paired_mask].copy()

        # Sort by UniqueID, pair_id, side, and PartID for consistent ordering
        sort_cols = ['UniqueID', 'pair_id', 'side', 'PartID']
        paired_gdf = paired_gdf.sort_values(sort_cols)

        # Assign segment_id within each pair_id group
        def assign_within_pair(pair_group):
            # Sort by PartID to ensure consistency
            pair_group = pair_group.sort_values(['side', 'PartID'])

            # For each pair, assign segment_id based on PartID order
            unique_parts = pair_group['PartID'].unique()
            part_to_segment = {part: i for i, part in enumerate(sorted(unique_parts))}

            pair_group['segment_id'] = pair_group['PartID'].map(part_to_segment)
            return pair_group

        paired_gdf = paired_gdf.groupby(['UniqueID', 'pair_id'], group_keys=False).apply(assign_within_pair)

        # Update the main dataframe
        gdf.loc[paired_mask, 'segment_id'] = paired_gdf['segment_id']

    # Process unpaired segments (pair_id == -1)
    unpaired_mask = gdf['pair_id'] == -1
    if unpaired_mask.any():
        # For unpaired segments, assign unique segment_ids within each UniqueID
        unpaired_gdf = gdf[unpaired_mask].copy()
        unpaired_gdf = unpaired_gdf.sort_values(['UniqueID', 'PartID'])

        def assign_unpaired_segments(unique_group):
            unique_group['segment_id'] = range(len(unique_group))
            return unique_group

        unpaired_gdf = unpaired_gdf.groupby('UniqueID', group_keys=False).apply(assign_unpaired_segments)
        gdf.loc[unpaired_mask, 'segment_id'] = unpaired_gdf['segment_id']

    return gdf


def update_path_with_suffix(input_path: str, suffix: str) -> str:
    """
    Update the input path to include a suffix before the extension.
    """
    if input_path.startswith("file://"):
        input_path = input_path[7:]
    dirname = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    if filename.endswith(".shp"):
        updated_filename = filename.replace(".shp", f"{suffix}.gpkg")
    else:
        updated_filename = filename.replace(".gpkg", f"{suffix}.gpkg")
    return os.path.join(dirname, updated_filename)


# ---------------------------
# Main processing function using Hydra
# ---------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up logging (default INFO level if not specified)
    log_level = cfg.get("logging", {"level": "INFO"}).get("level", "INFO")
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Get spacing from config (used in filename from split_to_plots)
    spacing = int(cfg.split_to_plots.perpendicular_spacing)

    # Build input path from split_to_plots output
    # This should be: {base}_footprint_ID_segments{spacing}m.gpkg
    footprint_base = update_path_with_suffix(cfg.dataset.ground_footprint, "_footprint_ID")
    input_path = footprint_base.replace(".gpkg", f"_segments{spacing}m.gpkg")

    # Build output path
    # This should be: {base}_footprint_ID_segments{spacing}m_sides.gpkg
    output_path = input_path.replace(".gpkg", "_sides.gpkg")

    logging.info(f"Input from split_to_plots: {input_path}")
    logging.info(f"Output for split_to_side: {output_path}")
    logging.info(f"Perpendicular spacing: {spacing}m")

    # Check if input file exists
    if not os.path.exists(input_path):
        logging.error(f"Input file does not exist: {input_path}")
        logging.error("Make sure split_to_plots has been run first!")
        return

    # Read input GeoDataFrame
    try:
        gdf = gpd.read_file(input_path)
        logging.info(f"Read {len(gdf)} segments from {input_path}")

        # ✅ CHECK: Verify plot_id exists from previous step
        if 'plot_id' in gdf.columns:
            n_unique_input = gdf['plot_id'].nunique()
            logging.info(f"✓ Input has {n_unique_input} unique plot_ids (will be preserved)")
        else:
            logging.warning("⚠ No plot_id column found in input! Creating from index...")
            gdf['plot_id'] = gdf.index

    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        return

    # Process segments to sort and find pairs
    logging.info("Processing segments for pairing...")
    paired_gdf = gdf.groupby("UniqueID", group_keys=False).apply(sort_segments_and_find_pairs)
    logging.info("Finished initial pairing.")

    # Assign segment_id within each pair_id (NOT plot_id!)
    logging.info("Assigning segment_id within pair groups...")
    paired_gdf = assign_segment_ids(paired_gdf)
    logging.info("Finished assigning segment_id.")

    # Optionally filter small polygons using min_area from configuration
    min_area = cfg.split_to_side.get("min_area", 5)
    initial_count = len(paired_gdf)
    paired_gdf = paired_gdf[paired_gdf.geometry.area >= min_area]
    filtered_count = initial_count - len(paired_gdf)
    logging.info(f"Filtered {filtered_count} small segments (< {min_area} m²)")
    logging.info(f"After filtering, {len(paired_gdf)} segments remain.")

    # ✅ VERIFY: Check that plot_id is still unique
    if 'plot_id' in paired_gdf.columns:
        n_features = len(paired_gdf)
        n_unique_plot_ids = paired_gdf['plot_id'].nunique()
        logging.info(f"✓ Output has {n_features} features with {n_unique_plot_ids} unique plot_ids")

        # For segments with sides, we expect 2 features per plot_id (left + right)
        ratio = n_features / n_unique_plot_ids if n_unique_plot_ids > 0 else 0
        logging.info(f"  Ratio: {ratio:.1f} features per plot_id (expected ~2.0 for paired segments)")

        # Sample
        sample_ids = paired_gdf['plot_id'].head(5).tolist()
        logging.info(f"  Sample plot_ids: {sample_ids}")

    # Log statistics about sides and pairs
    side_0_count = len(paired_gdf[paired_gdf['side'] == 0])
    side_1_count = len(paired_gdf[paired_gdf['side'] == 1])
    paired_count = len(paired_gdf[paired_gdf['pair_id'] >= 0])
    unpaired_count = len(paired_gdf[paired_gdf['pair_id'] == -1])

    logging.info(f"Side 0: {side_0_count} segments")
    logging.info(f"Side 1: {side_1_count} segments")
    logging.info(f"Paired segments: {paired_count}")
    logging.info(f"Unpaired segments: {unpaired_count}")

    # Log segment_id statistics
    if 'segment_id' in paired_gdf.columns:
        valid_segment_ids = paired_gdf[paired_gdf['segment_id'] >= 0]
        logging.info(f"Segments with valid segment_id: {len(valid_segment_ids)}")

        # Show segment_id distribution per pair
        if len(valid_segment_ids) > 0 and 'pair_id' in valid_segment_ids.columns:
            segments_per_pair = valid_segment_ids.groupby('pair_id')['segment_id'].max() + 1
            logging.info(
                f"Segments per pair - Mean: {segments_per_pair.mean():.1f}, Median: {segments_per_pair.median():.0f}, Max: {segments_per_pair.max()}")

    # Save the paired segments GeoDataFrame
    paired_gdf.to_file(output_path, driver="GPKG")
    logging.info(f"Paired segments saved to: {output_path}")
    logging.info("Split to sides complete!")


if __name__ == "__main__":
    main()