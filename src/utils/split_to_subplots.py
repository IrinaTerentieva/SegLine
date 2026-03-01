import os
import math
import logging
import pandas as pd
import geopandas as gpd
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

warnings.filterwarnings('ignore')


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


def generate_perpendiculars(centerline, avg_width, target_area, max_splitter_length=10):
    """
    Generate perpendicular lines to the centerline at intervals calculated to achieve a target area.
    """
    if avg_width <= 0:
        avg_width = 5
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
    (idx, footprint_row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id,
     target_area, extension_distance, width_column, crs) = args

    try:
        # Reconstruct geometry from WKT
        from shapely import wkt
        polygon = wkt.loads(geometry_wkt)

        unique_id = footprint_row_dict["UniqueID"]

        # Get width column name from config (passed as parameter)
        avg_width = footprint_row_dict.get(width_column, 0)

        max_width = avg_width + 10

        if max_width <= 5:
            max_width = 15
        if avg_width >= 9:
            target_area = int(target_area * 2)

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
        perpendiculars = generate_perpendiculars(extended_smooth_centerline, avg_width,
                                                 target_area, max_splitter_length=max_width)

        # Fall back to regular centerline if needed
        if len(perpendiculars) < 5:
            logging.debug(f'Bad perpendiculars for {unique_id}, switching to regular centerline')
            perpendiculars = generate_perpendiculars(extended_centerline, avg_width,
                                                     target_area, max_splitter_length=max_width)

        # Input polygons are already split into sides by split_to_sides,
        # so skip centerline splitting and only split by perpendiculars.
        segments = [polygon]
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


def process_subplots_parallel_optimized(footprint_gdf, centerline_gdf, smooth_centerline_gdf,
                                        target_area, output_path, extension_distance, width_column,
                                        plot_area, min_area, max_workers=None):
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

            args = (idx, row_dict, geometry_wkt, centerlines_by_id, smooth_centerlines_by_id,
                    target_area, extension_distance, width_column, footprint_gdf.crs)
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

        # Calculate areas and lengths
        split_polygons_gdf['area'] = split_polygons_gdf.geometry.area.round(1)
        split_polygons_gdf['length'] = split_polygons_gdf.geometry.length.round(1)
        split_polygons_gdf['perimeter'] = split_polygons_gdf.geometry.length.round(1)
        split_polygons_gdf['subplot_area'] = target_area
        split_polygons_gdf['plot_area'] = plot_area

        # Drop temporary columns and reset index
        if 'original_idx' in split_polygons_gdf.columns:
            split_polygons_gdf = split_polygons_gdf.drop(columns=['original_idx'])

        split_polygons_gdf = split_polygons_gdf.reset_index(drop=True)
        split_polygons_gdf['subplot_id'] = split_polygons_gdf.index

        # ✅ MERGE small subplots into adjacent neighbor on the same side
        n_before = len(split_polygons_gdf)
        side_col = 'side' if 'side' in split_polygons_gdf.columns else None
        group_cols = ['UniqueID', side_col] if side_col else ['UniqueID']
        group_cols = [c for c in group_cols if c is not None]

        merged_groups = []
        for group_key, group in split_polygons_gdf.groupby(group_cols):
            group = group.copy().reset_index(drop=True)
            changed = True
            while changed and len(group) > 1:
                changed = False
                for i in range(len(group)):
                    if group.at[i, 'area'] >= min_area:
                        continue
                    small_geom = group.at[i, 'geometry']
                    best_j = None
                    best_shared = 0.0
                    for j in range(len(group)):
                        if i == j:
                            continue
                        other_geom = group.at[j, 'geometry']
                        if not small_geom.buffer(0.01).intersects(other_geom):
                            continue
                        shared = small_geom.boundary.intersection(other_geom.boundary)
                        shared_len = shared.length if not shared.is_empty else 0
                        if shared_len == 0 and small_geom.distance(other_geom) < 0.1:
                            shared_len = 0.01
                        if shared_len > best_shared:
                            best_shared = shared_len
                            best_j = j
                    if best_j is not None:
                        group.at[best_j, 'geometry'] = unary_union(
                            [group.at[best_j, 'geometry'], small_geom])
                        group.at[best_j, 'area'] = round(group.at[best_j, 'geometry'].area, 1)
                        group = group.drop(i).reset_index(drop=True)
                        changed = True
                        logging.debug(f"Merged small subplot (area={small_geom.area:.1f}) in group {group_key}")
                        break
            merged_groups.append(group)

        split_polygons_gdf = gpd.GeoDataFrame(
            pd.concat(merged_groups, ignore_index=True), crs=footprint_gdf.crs)
        n_merged = n_before - len(split_polygons_gdf)
        if n_merged > 0:
            logging.info(f"Merged {n_merged} small subplots (area < {min_area:.1f} m²) into adjacent neighbors")

        # Recalculate after merging
        split_polygons_gdf['area'] = split_polygons_gdf.geometry.area.round(1)
        split_polygons_gdf['length'] = split_polygons_gdf.geometry.length.round(1)
        split_polygons_gdf['perimeter'] = split_polygons_gdf.geometry.length.round(1)

        # Reassign subplot_id after merging
        split_polygons_gdf = split_polygons_gdf.reset_index(drop=True)
        split_polygons_gdf['subplot_id'] = split_polygons_gdf.index

        # Debug: show remaining small subplots that were NOT merged
        still_small = split_polygons_gdf[split_polygons_gdf.geometry.area < min_area]
        if len(still_small) > 0:
            logging.warning(f"⚠ {len(still_small)} subplots still below min_area ({min_area:.1f} m²) after merge:")
            for idx, row in still_small.iterrows():
                geom = row.geometry
                uid = row.get('UniqueID', '?')
                side = row.get('side', '?')
                pid = row.get('plot_id', '?')
                same_group = split_polygons_gdf[
                    (split_polygons_gdf['UniqueID'] == uid) &
                    (split_polygons_gdf['side'] == side if 'side' in split_polygons_gdf.columns else True)
                ]
                same_group = same_group[same_group.index != idx]
                if len(same_group) == 0:
                    logging.warning(f"  fid={idx} | UniqueID={uid} side={side} plot_id={pid} "
                                    f"area={geom.area:.2f} | NO neighbors on same side")
                else:
                    distances = same_group.geometry.apply(lambda g: geom.distance(g))
                    nearest_idx = distances.idxmin()
                    nearest_dist = distances.min()
                    nearest_geom = same_group.at[nearest_idx, 'geometry']
                    shared = geom.boundary.intersection(nearest_geom.boundary)
                    shared_len = shared.length if not shared.is_empty else 0
                    touches = geom.touches(nearest_geom)
                    intersects = geom.intersects(nearest_geom)
                    buffered_intersects = geom.buffer(0.01).intersects(nearest_geom)
                    logging.warning(f"  fid={idx} | UniqueID={uid} side={side} plot_id={pid} "
                                    f"area={geom.area:.2f} | nearest_dist={nearest_dist:.4f} "
                                    f"shared_len={shared_len:.4f} touches={touches} "
                                    f"intersects={intersects} buffered={buffered_intersects}")

        # Save results
        split_polygons_gdf.to_file(output_path, driver="GPKG")
        logging.info(f"Split polygons saved to: {output_path}")
        logging.info(f"Created {len(split_polygons_gdf)} subplots from {len(footprint_gdf)} polygons")

        # Log statistics
        avg_area = split_polygons_gdf['area'].mean()
        median_area = split_polygons_gdf['area'].median()
        logging.info(f"Average subplot area: {avg_area:.2f} m²")
        logging.info(f"Median subplot area: {median_area:.2f} m²")
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
    num_workers = cfg.split_to_subplots.get("num_workers", None)
    segment_area = int(cfg.split_to_subplots.subplot_area)
    plot_area = float(cfg.split_to_plots.plot_area)
    min_area_fraction = float(cfg.split_to_subplots.min_area_fraction)
    min_area = min_area_fraction * plot_area
    extension_distance = cfg.split_to_plots.extension_distance
    max_splitter_length = cfg.split_to_subplots.max_splitter_length_buffer

    # Output path
    output_dir = os.path.dirname(footprint_path)
    output_filename = os.path.basename(footprint_path).replace("_sides.gpkg", f"_subplots{segment_area}m2.gpkg")
    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Input footprint (sides) path: {footprint_path}")
    logging.info(f"Regular centerline path: {centerline_path}")
    logging.info(f"Smooth centerline path: {smooth_centerline_path}")
    logging.info(f"Output path: {output_path}")
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
    logging.info(f"Merge threshold: subplots < {min_area:.1f} m² (min_area_fraction={min_area_fraction} * plot_area={plot_area}) will be merged")

    process_subplots_parallel_optimized(
        footprint_gdf,
        centerline_gdf,
        smooth_centerline_gdf,
        segment_area,
        output_path,
        extension_distance=extension_distance,
        width_column=cfg.dataset.width_column,
        plot_area=plot_area,
        min_area=min_area,
        max_workers=num_workers
    )

    logging.info("Subplot processing complete!")


if __name__ == "__main__":
    main()