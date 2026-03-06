import os
import glob
import math
import logging
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiLineString, GeometryCollection
from shapely.ops import split, linemerge, nearest_points
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf


# -----------------------------
# Geometry helper functions
# -----------------------------
def snap_linestring_gaps(geom, tolerance=1.0):
    """
    Snap small gaps in a MultiLineString by connecting endpoints that are within tolerance.
    Returns a LineString if all gaps are closed, or MultiLineString with fewer parts.
    """
    if not isinstance(geom, MultiLineString):
        return geom

    parts = list(geom.geoms)
    if len(parts) <= 1:
        return geom

    # Build a list of mutable coordinate lists
    chains = [list(p.coords) for p in parts]

    # Iteratively try to merge chains by snapping close endpoints
    changed = True
    while changed:
        changed = False
        for i in range(len(chains)):
            if chains[i] is None:
                continue
            for j in range(i + 1, len(chains)):
                if chains[j] is None:
                    continue
                # Check all 4 endpoint combinations
                pairs = [
                    ('end_i', 'start_j', chains[i][-1], chains[j][0]),
                    ('end_i', 'end_j', chains[i][-1], chains[j][-1]),
                    ('start_i', 'start_j', chains[i][0], chains[j][0]),
                    ('start_i', 'end_j', chains[i][0], chains[j][-1]),
                ]
                for label, _, p1, p2 in pairs:
                    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    if dist <= tolerance and dist > 0:
                        if label == 'end_i':
                            if _ == 'start_j':
                                chains[i] = chains[i] + chains[j]
                            else:
                                chains[i] = chains[i] + list(reversed(chains[j]))
                        else:
                            if _ == 'start_j':
                                chains[i] = list(reversed(chains[j])) + chains[i]
                            else:
                                chains[i] = chains[j] + chains[i]
                        chains[j] = None
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    remaining = [LineString(c) for c in chains if c is not None and len(c) >= 2]
    if len(remaining) == 1:
        return remaining[0]
    elif len(remaining) > 1:
        return linemerge(MultiLineString(remaining))
    return geom


def get_centerline_for_uid(centerline_gdf, uid):
    """
    Get the centerline geometry for a UniqueID.
    Snaps small gaps, merges parts, and returns the longest line if still fragmented.
    """
    cl_rows = centerline_gdf[centerline_gdf['UniqueID'] == uid]
    if cl_rows.empty:
        return None
    geom = cl_rows.geometry.iloc[0]
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        # Try snapping small gaps (< 1m)
        merged = snap_linestring_gaps(merged, tolerance=1.0)
        if isinstance(merged, MultiLineString):
            # Still fragmented — pick the longest part
            parts = list(merged.geoms)
            geom = max(parts, key=lambda g: g.length)
        else:
            geom = merged
    return geom


def assign_sides_spatially(subset, centerline_geom):
    """
    Assign side 0/1 based on which side of the centerline each polygon centroid falls on.
    Uses the signed distance (cross product) from the centerline direction.
    """
    if centerline_geom is None or centerline_geom.is_empty:
        # Fallback: use the naive half-split
        logging.warning("No centerline available, falling back to naive half-split")
        subset["side"] = 1
        half_rows = len(subset) // 2
        subset.iloc[half_rows:, subset.columns.get_loc("side")] = 0
        return subset

    # Get the overall direction vector of the centerline
    coords = list(centerline_geom.coords)
    start = np.array(coords[0])
    end = np.array(coords[-1])
    direction = end - start

    sides = []
    for _, row in subset.iterrows():
        centroid = row.geometry.centroid
        # Project centroid onto centerline, get nearest point
        nearest_pt = centerline_geom.interpolate(centerline_geom.project(Point(centroid.x, centroid.y)))
        # Cross product of direction vector with vector from nearest point to centroid
        to_centroid = np.array([centroid.x - nearest_pt.x, centroid.y - nearest_pt.y])
        cross = direction[0] * to_centroid[1] - direction[1] * to_centroid[0]
        sides.append(1 if cross >= 0 else 0)

    subset["side"] = sides
    return subset


def get_edge_points(polygon, precision=2):
    """
    Extract all edge points from a polygon's exterior.
    Using precision=2 (1cm) to handle small gaps from fragmented centerlines.
    """
    if polygon.is_empty or not polygon.is_valid:
        return set()
    edge_coords = polygon.exterior.coords
    edge_points = {(round(coord[0], precision), round(coord[1], precision)) for coord in edge_coords}
    return edge_points


def sort_segments_and_find_pairs(gdf, centerline_gdf=None):
    """
    Assign sides spatially using centerline, then find pairs based on shared edge points.
    Falls back to proximity-based pairing for polygons near centerline gaps.
    Returns a GeoDataFrame with "side" and "plot_id" fields.
    """

    def process_unique_id(subset):
        uid = subset['UniqueID'].iloc[0]

        # Get centerline for spatial side assignment
        centerline_geom = None
        if centerline_gdf is not None:
            centerline_geom = get_centerline_for_uid(centerline_gdf, uid)

        subset = assign_sides_spatially(subset, centerline_geom)
        subset["edge_points"] = subset.geometry.apply(get_edge_points)
        subset["plot_id"] = -1
        segment_id = 0
        side_0 = subset[subset["side"] == 0]
        side_1 = subset[subset["side"] == 1]

        # Track which polygons have been paired already
        paired = set()

        # Pass 1: exact edge-point matching (shared coordinates)
        min_area = 5  # minimum area to consider for pairing
        for idx_0, row_0 in side_0.iterrows():
            if idx_0 in paired or row_0.geometry.area < min_area:
                continue
            best_match = None
            best_shared = 0
            for idx_1, row_1 in side_1.iterrows():
                if idx_1 in paired or row_1.geometry.area < min_area:
                    continue
                shared_points = row_0["edge_points"].intersection(row_1["edge_points"])
                if len(shared_points) >= 2 and len(shared_points) > best_shared:
                    best_shared = len(shared_points)
                    best_match = idx_1
            if best_match is not None:
                subset.at[idx_0, "plot_id"] = segment_id
                subset.at[best_match, "plot_id"] = segment_id
                paired.add(idx_0)
                paired.add(best_match)
                segment_id += 1

        # Pass 2: proximity-based pairing for remaining large polygons
        # This handles gaps from fragmented centerlines where edge points don't match
        unpaired_0 = [i for i in side_0.index if i not in paired and side_0.loc[i].geometry.area >= min_area]
        unpaired_1 = [i for i in side_1.index if i not in paired and side_1.loc[i].geometry.area >= min_area]

        if unpaired_0 and unpaired_1:
            # Project each polygon's centroid onto the centerline to get along-corridor position
            if centerline_geom is not None and not centerline_geom.is_empty:
                for idx in unpaired_0:
                    c = subset.loc[idx].geometry.centroid
                    subset.at[idx, '_proj_dist'] = centerline_geom.project(Point(c.x, c.y))
                for idx in unpaired_1:
                    c = subset.loc[idx].geometry.centroid
                    subset.at[idx, '_proj_dist'] = centerline_geom.project(Point(c.x, c.y))

                # Match by closest projection distance along the centerline
                remaining_1 = set(unpaired_1)
                for idx_0 in unpaired_0:
                    if not remaining_1:
                        break
                    dist_0 = subset.at[idx_0, '_proj_dist']
                    best_match = None
                    best_diff = float('inf')
                    for idx_1 in remaining_1:
                        diff = abs(subset.at[idx_1, '_proj_dist'] - dist_0)
                        if diff < best_diff:
                            best_diff = diff
                            best_match = idx_1
                    # Only pair if they're within reasonable distance along the corridor
                    if best_match is not None and best_diff < 5.0:
                        subset.at[idx_0, "plot_id"] = segment_id
                        subset.at[best_match, "plot_id"] = segment_id
                        paired.add(idx_0)
                        paired.add(best_match)
                        remaining_1.discard(best_match)
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


def drop_sitetype(gdf):
    """Issue 7: Remove SiteType column if present."""
    if 'SiteType' in gdf.columns:
        logging.info("Dropping 'SiteType' column.")
        gdf = gdf.drop(columns=['SiteType'])
    return gdf


# ---------------------------
# Main processing function using Hydra
# ---------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up logging (default INFO level if not specified)
    log_level = cfg.get("logging", {"level": "INFO"}).get("level", "INFO")
    logging.basicConfig(level=log_level)
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Get splitting method
    splitting_method = cfg.split_to_side.splitting_method

    # Use the UniqueID output from the previous split_to_plots step.
    # Output directory will be the same as the input file directory.
    output_dir = os.path.dirname(cfg.dataset.ground_footprint)
    footprint_path = update_path_with_id(cfg.dataset.ground_footprint, output_dir)

    # Build the input file path for segments (glob to find real area in filename).
    base_name = os.path.basename(footprint_path).replace("_ID.gpkg", "")

    if splitting_method == "length":
        pattern = os.path.join(output_dir, f"{base_name}_footprint_ID_segments*m.gpkg")
        matches = [m for m in glob.glob(pattern) if 'm2.gpkg' not in m]
        if not matches:
            raise FileNotFoundError(f"No segments file found matching: {pattern}")
        input_filename = os.path.basename(sorted(matches)[-1])

    elif splitting_method == "area":
        pattern = os.path.join(output_dir, f"{base_name}_footprint_ID_segments*m2.gpkg")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No segments file found matching: {pattern}")
        input_filename = os.path.basename(sorted(matches)[-1])

    output_filename = base_name + "_sides.gpkg"


    # Build the input and output paths
    input_path = os.path.join(output_dir, input_filename)
    print('Working with: ', input_filename)
    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Footprint splitting method: {splitting_method}")
    logging.info(f"Input for split_to_side: {input_path}")
    logging.info(f"Output for split_to_side: {output_path}")
    if splitting_method == "length":
        logging.info(f"Segments file: {input_filename}")
    elif splitting_method == "area":
        logging.info(f"Segments file: {input_filename}")

    # Read input GeoDataFrame
    gdf = gpd.read_file(input_path)
    logging.info(f"Read {len(gdf)} segments from {input_path}")

    # Read centerline for spatial side assignment
    centerline_path = cfg.dataset.centerline
    if centerline_path.startswith("file://"):
        centerline_path = centerline_path[7:]
    centerline_gdf = gpd.read_file(centerline_path)
    # Ensure UniqueID is present on centerlines (map from SegID_new if needed)
    if 'UniqueID' not in centerline_gdf.columns and 'SegID_new' in centerline_gdf.columns:
        seg_to_uid = gdf[['SegID_new', 'UniqueID']].drop_duplicates()
        centerline_gdf = centerline_gdf.merge(seg_to_uid, on='SegID_new', how='left')
    logging.info(f"Read {len(centerline_gdf)} centerlines for spatial side assignment")

    # Process segments to sort and find pairs.
    paired_gdf = sort_segments_and_find_pairs(gdf, centerline_gdf=centerline_gdf)
    logging.info("Finished processing segments for pairing.")

    # Issue 2: Map (UniqueID, local_plot_id) to globally unique integer plot_id
    global_id = 0
    id_map = {}
    new_plot_ids = []
    for _, row in paired_gdf.iterrows():
        local_pid = row['plot_id']
        if local_pid == -1:
            new_plot_ids.append(-1)
            continue
        key = (row['UniqueID'], local_pid)
        if key not in id_map:
            id_map[key] = global_id
            global_id += 1
        new_plot_ids.append(id_map[key])
    paired_gdf['plot_id'] = new_plot_ids

    # Pass all polygons through -- small ones will be merged in split_to_subplots
    logging.info(f"Total segments: {len(paired_gdf)}")

    # Issue 7: Drop SiteType
    paired_gdf = drop_sitetype(paired_gdf)

    # Cleanup: Drop temp columns before saving
    temp_cols = ['edge_points', 'centroid_x', 'centroid_y', '_proj_dist']
    cols_to_drop = [c for c in temp_cols if c in paired_gdf.columns]
    if cols_to_drop:
        logging.info(f"Dropping temp columns: {cols_to_drop}")
        paired_gdf = paired_gdf.drop(columns=cols_to_drop)

    # Save the paired segments GeoDataFrame
    paired_gdf.to_file(output_path, driver="GPKG")
    print(f"Paired segments saved to: {output_path}")


if __name__ == "__main__":
    main()
