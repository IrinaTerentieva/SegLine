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
    Uses the LOCAL centerline direction at the nearest point (handles bends correctly).
    """
    if centerline_geom is None or centerline_geom.is_empty:
        # Fallback: use the naive half-split
        logging.warning("No centerline available, falling back to naive half-split")
        subset["side"] = 1
        half_rows = len(subset) // 2
        subset.iloc[half_rows:, subset.columns.get_loc("side")] = 0
        return subset

    cl_length = centerline_geom.length

    sides = []
    for _, row in subset.iterrows():
        centroid = row.geometry.centroid
        proj_dist = centerline_geom.project(Point(centroid.x, centroid.y))
        nearest_pt = centerline_geom.interpolate(proj_dist)

        # Local direction: tangent vector at the projected point
        delta = max(0.5, cl_length * 0.001)
        pt_before = centerline_geom.interpolate(max(0, proj_dist - delta))
        pt_after = centerline_geom.interpolate(min(cl_length, proj_dist + delta))
        direction = np.array([pt_after.x - pt_before.x, pt_after.y - pt_before.y])

        to_centroid = np.array([centroid.x - nearest_pt.x, centroid.y - nearest_pt.y])
        cross = direction[0] * to_centroid[1] - direction[1] * to_centroid[0]
        sides.append(1 if cross >= 0 else 0)

    subset["side"] = sides

    # Bend correction: at sharp bends, both strips may get the same side because
    # their centroids are on the same side of the local tangent. Detect same-side
    # pairs that share a significant linear boundary (the centerline between them)
    # and flip one to the opposite side.
    # Collect all candidate flip pairs first, then process once per part.
    indices = list(subset.index)
    flip_candidates = []  # (idx_to_flip, idx_partner, shared_length)
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i, idx_j = indices[i], indices[j]
            if subset.at[idx_i, 'side'] != subset.at[idx_j, 'side']:
                continue
            geom_i = subset.at[idx_i, 'geometry']
            geom_j = subset.at[idx_j, 'geometry']
            shared = geom_i.boundary.intersection(geom_j.boundary)
            # Threshold must be high enough to distinguish centerline boundaries (~30m)
            # from perpendicular cut boundaries (~1.5m corridor half-width)
            if shared.is_empty or shared.length < 5.0:
                continue
            # These two same-side parts share a significant boundary (the centerline).
            # The one closer to the centerline should be flipped.
            ci = geom_i.centroid
            cj = geom_j.centroid
            di = centerline_geom.distance(Point(ci.x, ci.y))
            dj = centerline_geom.distance(Point(cj.x, cj.y))
            flip_idx = idx_i if di < dj else idx_j
            flip_candidates.append((flip_idx, shared.length))

    # Apply flips, only once per part (highest shared length wins)
    flipped = set()
    # Sort by shared length descending for strongest signal first
    flip_candidates.sort(key=lambda x: -x[1])
    for flip_idx, slen in flip_candidates:
        if flip_idx in flipped:
            continue
        old_side = subset.at[flip_idx, 'side']
        subset.at[flip_idx, 'side'] = 1 - old_side
        flipped.add(flip_idx)
        logging.info(f"Bend correction: flipped PartID={subset.at[flip_idx, 'PartID']} "
                     f"from side {old_side} to {1 - old_side}")

    return subset


def get_edge_points(polygon, precision=2):
    """
    Extract all edge points from a polygon's exterior.
    Using precision=2 (1cm) to handle small gaps from fragmented centerlines.
    """
    if polygon.is_empty or not polygon.is_valid:
        return set()
    edge_points = set()
    if hasattr(polygon, 'geoms'):
        # MultiPolygon: collect edge points from all parts
        for part in polygon.geoms:
            edge_points.update(
                (round(c[0], precision), round(c[1], precision)) for c in part.exterior.coords
            )
    else:
        edge_points = {(round(c[0], precision), round(c[1], precision)) for c in polygon.exterior.coords}
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

        # Pass 1: edge-point matching — score all candidate pairs, then assign best-first
        candidates = []
        for idx_0, row_0 in side_0.iterrows():
            for idx_1, row_1 in side_1.iterrows():
                shared_points = row_0["edge_points"].intersection(row_1["edge_points"])
                if len(shared_points) >= 2:
                    candidates.append((len(shared_points), idx_0, idx_1))
        # Sort by shared points descending — strongest matches first
        candidates.sort(key=lambda x: -x[0])
        for n_shared, idx_0, idx_1 in candidates:
            if idx_0 in paired or idx_1 in paired:
                continue
            subset.at[idx_0, "plot_id"] = segment_id
            subset.at[idx_1, "plot_id"] = segment_id
            paired.add(idx_0)
            paired.add(idx_1)
            segment_id += 1

        # Pass 2: proximity-based pairing for remaining polygons
        # This handles gaps from fragmented centerlines where edge points don't match
        unpaired_0 = [i for i in side_0.index if i not in paired]
        unpaired_1 = [i for i in side_1.index if i not in paired]

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

        # Pass 3a: pair remaining unpaired parts with each other (cross-side, Euclidean distance)
        still_unpaired_0 = [i for i in subset.index
                            if subset.at[i, 'plot_id'] == -1 and subset.at[i, 'side'] == 0]
        still_unpaired_1 = [i for i in subset.index
                            if subset.at[i, 'plot_id'] == -1 and subset.at[i, 'side'] == 1]

        if still_unpaired_0 and still_unpaired_1:
            # Compute projection distances for filtering
            if centerline_geom is not None and not centerline_geom.is_empty:
                for idx in still_unpaired_0 + still_unpaired_1:
                    if '_proj_dist' not in subset.columns or np.isnan(subset.at[idx, '_proj_dist']):
                        c = subset.loc[idx].geometry.centroid
                        subset.at[idx, '_proj_dist'] = centerline_geom.project(Point(c.x, c.y))
            remaining_1 = set(still_unpaired_1)
            for idx_0 in still_unpaired_0:
                if not remaining_1:
                    break
                geom_0 = subset.loc[idx_0].geometry
                best_match = min(remaining_1, key=lambda i: geom_0.distance(subset.loc[i].geometry))
                best_dist = geom_0.distance(subset.loc[best_match].geometry)
                # Also check projection distance to avoid pairing across bends
                proj_ok = True
                if '_proj_dist' in subset.columns:
                    proj_diff = abs(subset.at[idx_0, '_proj_dist'] - subset.at[best_match, '_proj_dist'])
                    proj_ok = proj_diff < 20.0
                if best_dist < 100.0 and proj_ok:
                    subset.at[idx_0, 'plot_id'] = segment_id
                    subset.at[best_match, 'plot_id'] = segment_id
                    paired.add(idx_0)
                    paired.add(best_match)
                    remaining_1.discard(best_match)
                    segment_id += 1

        # Pass 3b: merge any still-unpaired parts into nearest paired neighbor on same side
        still_unpaired = [i for i in subset.index if subset.at[i, 'plot_id'] == -1]
        if still_unpaired:
            for idx in still_unpaired:
                side = subset.at[idx, 'side']
                geom = subset.at[idx, 'geometry']
                # Find nearest paired part on the same side
                same_side_paired = subset[(subset['side'] == side) & (subset['plot_id'] != -1)]
                if same_side_paired.empty:
                    continue
                best_idx = None
                best_shared = 0
                for cidx, crow in same_side_paired.iterrows():
                    shared = geom.boundary.intersection(crow.geometry.boundary)
                    slen = shared.length if not shared.is_empty else 0
                    if slen > best_shared:
                        best_shared = slen
                        best_idx = cidx
                # Fallback to nearest by distance if no shared boundary
                if best_idx is None:
                    dists = same_side_paired.geometry.distance(geom)
                    best_idx = dists.idxmin()
                subset.at[idx, 'plot_id'] = subset.at[best_idx, 'plot_id']

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

    # Read cleaned centerline (with UniqueID) for spatial side assignment
    cl_raw_path = cfg.dataset.centerline
    if cl_raw_path.startswith("file://"):
        cl_raw_path = cl_raw_path[7:]
    cl_dir = os.path.dirname(cl_raw_path)
    cl_base = os.path.basename(cl_raw_path).replace(".gpkg", "_centerline_ID.gpkg")
    centerline_id_path = os.path.join(cl_dir, cl_base)
    if not os.path.exists(centerline_id_path):
        logging.warning(f"Cleaned centerline not found, using raw: {cl_raw_path}")
        centerline_id_path = cl_raw_path
    centerline_gdf = gpd.read_file(centerline_id_path)
    # Ensure UniqueID is present on centerlines (map from SegID_new if needed)
    if 'UniqueID' not in centerline_gdf.columns and 'SegID_new' in centerline_gdf.columns:
        seg_to_uid = gdf[['SegID_new', 'UniqueID']].drop_duplicates()
        centerline_gdf = centerline_gdf.merge(seg_to_uid, on='SegID_new', how='left')
    logging.info(f"Read {len(centerline_gdf)} centerlines from {centerline_id_path}")

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
