import os
import math
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon, GeometryCollection, Point
from shapely.ops import split, linemerge, substring, unary_union
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings
from shapely.geometry import Point

warnings.filterwarnings('ignore')

# -----------------------------
# Identify subplot pairs
# -----------------------------
def get_edge_points(polygon, precision=3):
    """
    Extract all edge points from a polygon's exterior.
    """
    if polygon.is_empty or not polygon.is_valid:
        return set()
    edge_coords = polygon.exterior.coords
    edge_points = {(round(coord[0], precision), round(coord[1], precision))
                   for coord in edge_coords}
    return edge_points


def find_subplot_pairs_edge_based(gdf):
    """
    Find pairs of subplots using edge point matching within plot groups.
    This approach is more robust than position-based matching as it directly
    identifies which subplots share boundaries from perpendicular cuts.
    """
    gdf = gdf.copy()
    gdf['pair_id'] = -1  # Initialize with -1 (unpaired)

    # Check if plot_id exists from split_to_sides
    if 'plot_id' not in gdf.columns:
        logging.warning("No plot_id column found. Using pure edge matching across all subplots.")
        return find_subplot_pairs_pure_edge_matching(gdf)

    global_pair_counter = 0

    # Process each plot_id group (these are already paired sides)
    unique_plots = gdf[gdf['plot_id'] >= 0]['plot_id'].unique()

    for plot_id in unique_plots:
        plot_group = gdf[gdf['plot_id'] == plot_id].copy()

        if len(plot_group) < 2:
            continue

        # Add edge points for all subplots in this group
        plot_group['edge_points'] = plot_group.geometry.apply(
            lambda g: get_edge_points(g, precision=3)
        )

        # Split by side (0 or 1)
        side_0 = plot_group[plot_group['side'] == 0]
        side_1 = plot_group[plot_group['side'] == 1]

        if side_0.empty or side_1.empty:
            continue

        # Track which subplots have been paired
        paired_indices = set()

        # For each subplot on side 0, find its match on side 1
        for idx_0, row_0 in side_0.iterrows():
            if idx_0 in paired_indices:
                continue

            if row_0.geometry.area < 5:  # Skip very small polygons
                continue

            best_match = None
            best_shared_points = 0

            # Check all subplots on the opposite side
            for idx_1, row_1 in side_1.iterrows():
                if idx_1 in paired_indices:
                    continue

                if row_1.geometry.area < 5:  # Skip very small polygons
                    continue

                # Find shared edge points
                shared_points = row_0['edge_points'].intersection(row_1['edge_points'])

                # We need at least 2 shared points to consider it a match
                # (2 points define the edge from the perpendicular cut)
                if len(shared_points) >= 2:
                    # If multiple candidates, choose the one with most shared points
                    if len(shared_points) > best_shared_points:
                        best_shared_points = len(shared_points)
                        best_match = idx_1

            # Create pair if match found
            if best_match is not None:
                gdf.loc[idx_0, 'pair_id'] = global_pair_counter
                gdf.loc[best_match, 'pair_id'] = global_pair_counter
                paired_indices.add(idx_0)
                paired_indices.add(best_match)
                global_pair_counter += 1

    return gdf


def find_subplot_pairs_pure_edge_matching(gdf):
    """
    Fallback method using pure edge matching when no plot_id information is available.
    Similar to the split_to_sides approach but applied to all subplots.
    """
    gdf = gdf.copy()
    gdf['pair_id'] = -1

    # Process each UniqueID (corridor) separately
    for unique_id in gdf['UniqueID'].unique():
        subset = gdf[gdf['UniqueID'] == unique_id].copy()

        if len(subset) < 2:
            continue

        # Calculate edge points for all subplots
        subset['edge_points'] = subset.geometry.apply(
            lambda g: get_edge_points(g, precision=3)
        )
        subset['centroid'] = subset.geometry.centroid
        subset['area'] = subset.geometry.area

        # Filter out very small polygons
        subset = subset[subset['area'] >= 5]

        if len(subset) < 2:
            continue

        # Determine corridor orientation to help identify sides
        all_centroids = subset['centroid']
        x_coords = [c.x for c in all_centroids]
        y_coords = [c.y for c in all_centroids]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        is_horizontal = x_range > y_range

        # Calculate median perpendicular position to identify sides
        if is_horizontal:
            median_y = np.median(y_coords)
            subset['side_indicator'] = subset['centroid'].apply(lambda c: 0 if c.y < median_y else 1)
        else:
            median_x = np.median(x_coords)
            subset['side_indicator'] = subset['centroid'].apply(lambda c: 0 if c.x < median_x else 1)

        pair_counter = 0
        paired_indices = set()

        # Process each subplot on side 0
        side_0 = subset[subset['side_indicator'] == 0]
        side_1 = subset[subset['side_indicator'] == 1]

        for idx_0, row_0 in side_0.iterrows():
            if idx_0 in paired_indices:
                continue

            best_match = None
            best_score = 0

            for idx_1, row_1 in side_1.iterrows():
                if idx_1 in paired_indices:
                    continue

                # Calculate shared edge points
                shared_points = row_0['edge_points'].intersection(row_1['edge_points'])

                if len(shared_points) >= 2:
                    # Score based on number of shared points and area similarity
                    area_ratio = min(row_0['area'], row_1['area']) / max(row_0['area'], row_1['area'])
                    score = len(shared_points) * (1 + area_ratio)

                    if score > best_score:
                        best_score = score
                        best_match = idx_1

            # Create pair if good match found
            if best_match is not None:
                gdf.loc[idx_0, 'pair_id'] = pair_counter
                gdf.loc[best_match, 'pair_id'] = pair_counter
                paired_indices.add(idx_0)
                paired_indices.add(best_match)
                pair_counter += 1

    return gdf


# Replace the existing find_subplot_pairs function in your split_to_subplots.py with this:
def find_subplot_pairs(gdf):
    """
    Main function to find subplot pairs - uses edge-based matching for reliability.
    """
    return find_subplot_pairs_edge_based(gdf)


# -----------------------------
# Geometry helper functions
# -----------------------------
def extend_line(line: LineString, extension_distance=100):
    """
    Extend a line at both ends by a given distance.
    Handles MultiLineString by merging.
    """
    if isinstance(line, MultiLineString):
        line = linemerge(line)
    if not isinstance(line, LineString) or line.is_empty:
        return line
    try:
        coords = list(line.coords)
        if len(coords) < 2:
            return line

        # Extend at the start
        start_x, start_y = coords[0]
        next_x, next_y = coords[1]
        dx, dy = start_x - next_x, start_y - next_y
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return line
        start_extension = (
            start_x + (dx / length) * extension_distance,
            start_y + (dy / length) * extension_distance,
        )

        # Extend at the end
        end_x, end_y = coords[-1]
        prev_x, prev_y = coords[-2]
        dx, dy = end_x - prev_x, end_y - prev_y
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return line
        end_extension = (
            end_x + (dx / length) * extension_distance,
            end_y + (dy / length) * extension_distance,
        )

        extended_coords = [start_extension] + coords + [end_extension]
        return LineString(extended_coords)
    except Exception as e:
        logging.debug(f"Error extending line: {e}")
        return line


def generate_perpendiculars(centerline, avg_width, splitting_method, target_area, target_length, max_splitter_length=10):
    """
    Generate perpendicular lines to the centerline at intervals calculated to achieve a target area.
    """
    if avg_width <= 0:
        avg_width = 5

    # Set spacing, either by width or by area
    if splitting_method == "length":
        spacing = target_length
    elif splitting_method == "area":
        spacing = target_area / avg_width

    if spacing <= 0 or centerline.length <= 0:
        return []

    perpendiculars = []
    try:
        for distance in np.arange(0, centerline.length, spacing):
            point = centerline.interpolate(distance)
            next_point = centerline.interpolate(min(distance + 1, centerline.length))
            dx, dy = next_point.x - point.x, next_point.y - point.y
            perpendicular_vector = (-dy, dx)
            length = np.sqrt(perpendicular_vector[0] ** 2 + perpendicular_vector[1] ** 2)
            if length == 0:
                continue
            unit_vector = (perpendicular_vector[0] / length, perpendicular_vector[1] / length)
            half_length = max_splitter_length / 2
            start = (point.x - unit_vector[0] * half_length,
                     point.y - unit_vector[1] * half_length)
            end = (point.x + unit_vector[0] * half_length,
                   point.y + unit_vector[1] * half_length)
            perpendicular_line = LineString([start, end])
            perpendiculars.append(perpendicular_line)
    except Exception as e:
        logging.debug(f"Error in generate_perpendiculars: {e}")
    return perpendiculars


def split_geometry(geometry, splitter):
    """
    Split a geometry with a splitter, handling GeometryCollection properly.
    """
    try:
        result = split(geometry, splitter)
        if isinstance(result, GeometryCollection):
            return [geom for geom in result.geoms if isinstance(geom, Polygon)]
        elif isinstance(result, Polygon):
            return [result]
        else:
            return []
    except Exception as e:
        logging.debug(f"Error splitting geometry: {e}")
        return []


def process_subplot_worker(args):
    """
    Worker function for multiprocessing. Unpacks arguments and processes a single polygon.
    """
    (idx, footprint_row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id, splitting_method,
     target_area, target_length, extension_distance, crs) = args

    try:
        # Reconstruct geometry from WKT
        from shapely import wkt
        polygon = wkt.loads(geometry_wkt)

        unique_id = footprint_row_dict["UniqueID"]
        avg_width = footprint_row_dict.get("avg_width", 0)
        max_width = avg_width + 10

        if max_width <= 5:
            max_width = 15
        if avg_width >= 9:
            target_area = int(target_area * 2)  # For wide lines, increase subplot area (area-based splitting only)

        # Get centerlines from dictionaries
        centerline_wkt = centerlines_by_id.get(unique_id)
        smooth_centerline_wkt = smooth_centerlines_by_id.get(unique_id)

        if not centerline_wkt or not smooth_centerline_wkt:
            logging.debug(f"No centerlines for UniqueID: {unique_id}")
            return []

        # Reconstruct geometries
        centerline_geom = wkt.loads(centerline_wkt)
        smooth_centerline_geom = wkt.loads(smooth_centerline_wkt)

        if isinstance(smooth_centerline_geom, MultiLineString):
            smooth_centerline_geom = linemerge(smooth_centerline_geom)
        if isinstance(centerline_geom, MultiLineString):
            centerline_geom = linemerge(centerline_geom)

        extended_centerline = extend_line(centerline_geom, extension_distance)
        extended_smooth_centerline = extend_line(smooth_centerline_geom, extension_distance)

        # Try smooth centerline first
        perpendiculars = generate_perpendiculars(extended_smooth_centerline, avg_width, splitting_method,
                                                 target_area, target_length, max_splitter_length=max_width)

        # Fall back to regular centerline if needed
        if len(perpendiculars) < 5:
            logging.debug(f'Bad perpendiculars for {unique_id}, switching to regular centerline')
            perpendiculars = generate_perpendiculars(extended_centerline, avg_width, splitting_method,
                                                     target_area, target_length, max_splitter_length=max_width)

        # Split polygon
        segments = split_geometry(polygon, extended_centerline)
        for perp in perpendiculars:
            temp_segments = []
            for segment in segments:
                temp_segments.extend(split_geometry(segment, perp))
            segments = temp_segments

        # Return results
        results = []
        for part_id, segment in enumerate(segments):
            if segment.area > 0:  # Only include valid segments
                result = footprint_row_dict.copy()
                result['geometry'] = segment.wkt  # Store as WKT for serialization
                result['PartID'] = part_id
                result['original_idx'] = idx
                results.append(result)

        return results

    except Exception as e:
        logging.error(f"Error processing subplot {idx}: {e}")
        return []


def process_subplots_parallel_optimized(footprint_gdf, centerline_gdf, smooth_centerline_gdf, splitting_method,
                                        target_area, target_length, output_path, extension_distance=50, max_workers=None):
    """
    Optimized parallel processing using multiprocessing.Pool.
    """
    if max_workers is None:
        max_workers = min(cpu_count(), 8)

    logging.info(f"Using {max_workers} workers for parallel processing")

    # Pre-process centerlines into dictionaries for faster lookup
    centerlines_by_id = {}
    smooth_centerlines_by_id = {}

    for _, row in centerline_gdf.iterrows():
        if row['UniqueID'] and row.geometry:
            centerlines_by_id[row['UniqueID']] = row.geometry.wkt

    for _, row in smooth_centerline_gdf.iterrows():
        if row['UniqueID'] and row.geometry:
            smooth_centerlines_by_id[row['UniqueID']] = row.geometry.wkt

    # Log statistics
    logging.info(f"Loaded {len(centerlines_by_id)} regular centerlines")
    logging.info(f"Loaded {len(smooth_centerlines_by_id)} smooth centerlines")

    # Prepare arguments for workers
    worker_args = []
    for idx, row in footprint_gdf.iterrows():
        if row.geometry and row.geometry.is_valid:
            # Convert row to dict without geometry
            row_dict = row.drop('geometry').to_dict()
            # Store geometry as WKT for serialization
            geometry_wkt = row.geometry.wkt

            args = (idx, row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id, splitting_method,
                    target_area, target_length, extension_distance, footprint_gdf.crs)
            worker_args.append(args)

    logging.info(f"Processing {len(worker_args)} polygons...")

    # Process in parallel
    results = []
    successful_count = 0

    with Pool(processes=max_workers) as pool:
        # Use imap for better memory efficiency and progress tracking
        for result_batch in tqdm(pool.imap(process_subplot_worker, worker_args, chunksize=10),
                                 total=len(worker_args), desc="Processing subplots"):
            if result_batch:
                results.extend(result_batch)
                successful_count += 1

    logging.info(f"Successfully processed {successful_count}/{len(worker_args)} polygons")

    # Convert results back to GeoDataFrame
    if results:
        from shapely import wkt

        # Convert WKT geometries back to shapely objects
        for result in results:
            result['geometry'] = wkt.loads(result['geometry'])

        split_polygons_gdf = gpd.GeoDataFrame(results, crs=footprint_gdf.crs)

        # Calculate areas
        split_polygons_gdf['area'] = split_polygons_gdf.geometry.area

        # Drop temporary columns and reset index
        if 'original_idx' in split_polygons_gdf.columns:
            split_polygons_gdf = split_polygons_gdf.drop(columns=['original_idx'])

        split_polygons_gdf = split_polygons_gdf.reset_index(drop=True)
        split_polygons_gdf['subplot_id'] = split_polygons_gdf.index

        # Clean empties/zero-area features
        split_polygons_gdf = clean_empty_geometries(split_polygons_gdf)

        # Find subplot pairs using subplot ordering relationships (along centerline) - NEW
        logging.info("Pairing subplots...")
        split_polygons_gdf = find_subplot_pairs(split_polygons_gdf)

        # Log pairing statistics
        paired = split_polygons_gdf[split_polygons_gdf['pair_id'] >= 0]
        unpaired = split_polygons_gdf[split_polygons_gdf['pair_id'] < 0]
        n_pairs = paired['pair_id'].nunique() if not paired.empty else 0

        logging.info(f"Pairing results: {len(paired)} subplots in {n_pairs} pairs, "
                     f"{len(unpaired)} unpaired")

        # Recompute area + subplot_id after cleaning for accurate stats/ids
        if not split_polygons_gdf.empty:
            split_polygons_gdf['area'] = split_polygons_gdf.geometry.area
            split_polygons_gdf = split_polygons_gdf.reset_index(drop=True)
            split_polygons_gdf['subplot_id'] = split_polygons_gdf.index

            # Save results
            split_polygons_gdf.to_file(output_path, driver="GPKG")
            logging.info(f"Split polygons saved to: {output_path}")
            logging.info(f"Created {len(split_polygons_gdf)} subplots from {len(footprint_gdf)} polygons")

            # Stats
            avg_area = float(split_polygons_gdf['area'].mean()) if len(split_polygons_gdf) else 0.0
            median_area = float(split_polygons_gdf['area'].median()) if len(split_polygons_gdf) else 0.0
            logging.info(f"Average subplot area: {avg_area:.2f} m²")
            logging.info(f"Median subplot area: {median_area:.2f} m²")
        else:
            logging.warning("All split features were removed during cleaning. Nothing to write.")
    else:
        logging.warning("No results generated from subplot splitting")


def plot_perpendiculars(centerline, perpendiculars, output_path):
    """
    Plot centerline and perpendiculars for debugging.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(centerline, LineString):
        x, y = centerline.xy
        ax.plot(x, y, label="Centerline", color="blue", linewidth=2)
    for perp in perpendiculars:
        if isinstance(perp, LineString):
            px, py = perp.xy
            ax.plot(px, py, color="red", linewidth=1)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Centerline and Perpendiculars")
    ax.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved to {output_path}")


def clean_empty_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove features with null/empty geometry or area <= 0 (after fixing simple invalids for area calc).
    Returns a new GeoDataFrame (does not modify input in place).
    """
    if gdf is None or gdf.empty:
        logging.info("No features to clean (empty GeoDataFrame).")
        return gdf

    initial = len(gdf)

    # Drop null/empty geometries
    mask_valid_geom = (~gdf.geometry.isna()) & (~gdf.geometry.is_empty)
    dropped_empty = int((~mask_valid_geom).sum())
    if dropped_empty:
        logging.info(f"Dropping {dropped_empty} features with empty/null geometry")
    gdf = gdf.loc[mask_valid_geom].copy()

    if gdf.empty:
        logging.info("All features were empty/null after geometry filter.")
        return gdf.reset_index(drop=True)

    # Ensure polygonal type only (defensive)
    poly_like = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    dropped_nonpoly = int((~poly_like).sum())
    if dropped_nonpoly:
        logging.info(f"Dropping {dropped_nonpoly} non-polygon geometries")
    gdf = gdf.loc[poly_like].copy()

    if gdf.empty:
        logging.info("No polygonal features remain after type filter.")
        return gdf.reset_index(drop=True)

    # Compute robust area for filtering (use buffer(0) only for area computation, not to overwrite geometry)
    try:
        tmp_area = gpd.GeoSeries(gdf.geometry, crs=gdf.crs).buffer(0).area
    except Exception:
        # Fallback without buffer if GEOS issues
        tmp_area = gdf.geometry.area

    area_round = np.round(tmp_area.to_numpy(), 6)  # higher precision; you can relax to 2 if desired
    positive_area = np.isfinite(area_round) & (area_round > 0.0)
    dropped_zero = int((~positive_area).sum())
    if dropped_zero:
        logging.info(f"Dropping {dropped_zero} features with area <= 0 m²")

    gdf = gdf.loc[positive_area].copy().reset_index(drop=True)

    logging.info(f"Cleaned features: {len(gdf)}/{initial} remain after empty/area filters")
    return gdf


# ----------------------------
# File reading helpers
# ----------------------------
def read_vector_file(path: str, layer: str = None) -> gpd.GeoDataFrame:
    """
    Reads a vector file with optional layer specification.
    """
    if path.startswith("file://"):
        path = path[7:]
    try:
        if layer:
            return gpd.read_file(path, layer=layer)
        else:
            return gpd.read_file(path)
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        raise


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
# Main processing function
# ---------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up logging
    log_level = cfg.get("logging", {"level": "INFO"}).get("level", "INFO")
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Determine paths
    footprint_path = update_path_with_suffix(cfg.dataset.ground_footprint, "_sides")

    # Get centerline paths based on configuration
    if cfg.split_to_subplots.get("use_smooth_centerline", True) and cfg.smoothening.get("perform_smoothing", True):
        centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")
        smooth_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID_smooth")
    else:
        # If no smoothing, use regular centerline for both
        centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")
        smooth_centerline_path = centerline_path

    # Configuration parameters
    splitting_method = cfg.params.splitting_method
    num_workers = cfg.split_to_subplots.get("num_workers", None)
    segment_area = int(cfg.split_to_subplots.segment_area * 2)  # NB - added *2 bc segment_area is for just one side (and there are 2)
    segment_length = int(cfg.split_to_subplots.segment_length)
    extension_distance = cfg.split_to_plots.extension_distance
    max_splitter_length = cfg.split_to_subplots.max_splitter_length_buffer

    # Output path
    output_dir = os.path.dirname(footprint_path)
    if splitting_method == "length":
        output_filename = os.path.basename(footprint_path).replace("_sides.gpkg", f"_subplots{segment_length}m.gpkg")
    if splitting_method == "area":
        output_filename = os.path.basename(footprint_path).replace("_sides.gpkg", f"_subplots{segment_area}m2.gpkg")

    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Footprint splitting method: {splitting_method}")
    logging.info(f"Input footprint (sides) path: {footprint_path}")
    logging.info(f"Regular centerline path: {centerline_path}")
    logging.info(f"Smooth centerline path: {smooth_centerline_path}")
    logging.info(f"Output path: {output_path}")
    if splitting_method == "length":
        logging.info(f"Parameters: segment_length={segment_length}m, extension={extension_distance}m")
    elif splitting_method == "area":
        logging.info(f"Parameters: segment_area={segment_area}m², extension={extension_distance}m")

    # Read input data
    try:
        logging.info("Reading footprint data...")
        footprint_gdf = read_vector_file(footprint_path)

        logging.info("Reading centerline data...")
        centerline_gdf = read_vector_file(centerline_path)
        smooth_centerline_gdf = read_vector_file(smooth_centerline_path)

        logging.info(f"Loaded {len(footprint_gdf)} footprints, {len(centerline_gdf)} centerlines")

    except Exception as e:
        logging.error(f"Failed to read input files: {e}")
        return

    # Process subplots
    process_subplots_parallel_optimized(
        footprint_gdf,
        centerline_gdf,
        smooth_centerline_gdf,
        splitting_method,
        segment_area,
        segment_length,
        output_path,
        extension_distance=extension_distance,
        max_workers=num_workers
    )

    logging.info("Subplot processing complete!")


if __name__ == "__main__":
    main()