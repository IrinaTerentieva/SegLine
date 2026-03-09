import os
import math
import logging
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
import pickle
import warnings

warnings.filterwarnings('ignore')


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

    chains = [list(p.coords) for p in parts]

    changed = True
    while changed:
        changed = False
        for i in range(len(chains)):
            if chains[i] is None:
                continue
            for j in range(i + 1, len(chains)):
                if chains[j] is None:
                    continue
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


def generate_perpendiculars(centerline, avg_width, target_area, max_splitter_length=10,
                            start_offset=0.0):
    """
    Generate perpendicular lines to the centerline at intervals calculated to achieve a target area.
    start_offset: distance along centerline where the actual corridor starts (for extended centerlines).
    Perpendiculars start at start_offset so leftover is only at the far end.
    Skips the last perpendicular if it would create a leftover fragment smaller than half the target.
    """
    if avg_width <= 0:
        avg_width = 5
    spacing = target_area / avg_width
    if spacing <= 0 or centerline.length <= 0:
        return []

    perpendiculars = []
    try:
        distances = list(np.arange(start_offset, centerline.length, spacing))

        # Drop the last perpendicular if it would leave a fragment < half the spacing
        if len(distances) > 1:
            remaining = centerline.length - distances[-1]
            if remaining < spacing * 0.5:
                distances = distances[:-1]

        for distance in distances:
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


def _same_side_of_centerline(geom_a, geom_b, centerline):
    """Check if two polygons' centroids are on the same side of the centerline.
    Uses local tangent at each centroid's projection point independently."""
    if centerline is None:
        return True
    from shapely.geometry import Point
    ca = geom_a.centroid
    cb = geom_b.centroid
    cl_len = centerline.length
    delta = max(0.5, cl_len * 0.001)

    def _cross(centroid, proj):
        pt_on = centerline.interpolate(proj)
        pt_before = centerline.interpolate(max(0, proj - delta))
        pt_after = centerline.interpolate(min(cl_len, proj + delta))
        dx = pt_after.x - pt_before.x
        dy = pt_after.y - pt_before.y
        return dx * (centroid.y - pt_on.y) - dy * (centroid.x - pt_on.x)

    cross_a = _cross(ca, centerline.project(Point(ca.x, ca.y)))
    cross_b = _cross(cb, centerline.project(Point(cb.x, cb.y)))
    return (cross_a >= 0) == (cross_b >= 0)


def merge_small_fragments(segments, min_area, centerline=None):
    """
    Merge small polygon fragments into their largest boundary-sharing neighbor.
    Only merges into neighbors on the same side of the centerline.
    Repeats until no small fragments remain or no merges are possible.
    """
    if len(segments) <= 1:
        return segments

    merged = list(segments)
    changed = True
    while changed:
        changed = False
        areas = [s.area for s in merged]
        # Process smallest fragments first
        order = sorted(range(len(merged)), key=lambda i: areas[i])
        for i in order:
            if areas[i] >= min_area:
                continue
            # Find the best neighbor to merge into
            best_j = None
            best_score = (-1, 0)  # (has_shared_edge, area)
            for j in range(len(merged)):
                if j == i:
                    continue
                # Check adjacency: touching, shared edge, or very close (float imprecision)
                is_neighbor = (merged[i].touches(merged[j])
                               or (not merged[i].intersection(merged[j]).is_empty
                                   and merged[i].intersection(merged[j]).length > 0)
                               or merged[i].distance(merged[j]) < 0.01)
                if not is_neighbor:
                    continue
                # Only merge into neighbors on the same side of the centerline
                if not _same_side_of_centerline(merged[i], merged[j], centerline):
                    continue
                # Prefer neighbors with shared edge (clean union) over point-touch
                inter = merged[i].boundary.intersection(merged[j].boundary)
                shared_len = inter.length if not inter.is_empty else 0
                score = (1 if shared_len > 0 else 0, areas[j])
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_j is not None:
                combined = unary_union([merged[best_j], merged[i]])
                # If union creates MultiPolygon, use progressively larger buffer to merge
                if hasattr(combined, 'geoms'):
                    for buf in [0.001, 0.01, 0.05]:
                        combined = combined.buffer(buf).buffer(-buf)
                        if not hasattr(combined, 'geoms'):
                            break
                    if hasattr(combined, 'geoms'):
                        combined = max(combined.geoms, key=lambda g: g.area)
                merged[best_j] = combined
                merged.pop(i)
                changed = True
                break  # Restart after each merge

    # Drop any remaining fragments below min_area that couldn't merge
    merged = [s for s in merged if s.area >= min_area]

    return merged


def process_polygon_worker(args):
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
            return []

        # Reconstruct geometries
        centerline_geom = wkt.loads(centerline_wkt)
        smooth_centerline_geom = wkt.loads(smooth_centerline_wkt)

        if isinstance(smooth_centerline_geom, MultiLineString):
            smooth_centerline_geom = linemerge(smooth_centerline_geom)
            if isinstance(smooth_centerline_geom, MultiLineString):
                smooth_centerline_geom = snap_linestring_gaps(smooth_centerline_geom, tolerance=1.0)
        if isinstance(centerline_geom, MultiLineString):
            centerline_geom = linemerge(centerline_geom)
            if isinstance(centerline_geom, MultiLineString):
                centerline_geom = snap_linestring_gaps(centerline_geom, tolerance=1.0)

        extended_centerline = extend_line(centerline_geom, extension_distance)
        extended_smooth_centerline = extend_line(smooth_centerline_geom, extension_distance)

        # Try smooth centerline first — start at corridor boundary (after extension)
        perpendiculars = generate_perpendiculars(extended_smooth_centerline, avg_width,
                                                 target_area, max_splitter_length=max_width,
                                                 start_offset=extension_distance)

        # Fall back to regular centerline if needed
        if len(perpendiculars) < 5:
            perpendiculars = generate_perpendiculars(extended_centerline, avg_width,
                                                     target_area, max_splitter_length=max_width,
                                                     start_offset=extension_distance)

        # Split polygon
        segments = split_geometry(polygon, extended_centerline)
        for perp in perpendiculars:
            temp_segments = []
            for segment in segments:
                temp_segments.extend(split_geometry(segment, perp))
            segments = temp_segments

        # Merge small fragments into adjacent neighbors (same side of centerline only)
        min_frag_area = target_area * 0.25  # Merge fragments < 25% of target (handles endpoint leftovers)
        if len(segments) > 1:
            segments = merge_small_fragments(segments, min_frag_area, centerline=centerline_geom)

        # Clean up near-degenerate edges (spikes from centerline split)
        segments = [s.simplify(0.01, preserve_topology=True) for s in segments]

        # Return results
        results = []
        for part_id, segment in enumerate(segments):
            result = footprint_row_dict.copy()
            result['geometry'] = segment.wkt  # Store as WKT for serialization
            result['PartID'] = part_id
            result['original_idx'] = idx
            results.append(result)

        return results

    except Exception as e:
        logging.error(f"Error processing polygon {idx}: {e}")
        return []


def build_centerline_dict(centerline_gdf):
    """
    Group centerlines by UniqueID and merge with linemerge(MultiLineString(...))
    instead of dict overwrite. Ensures all centerline parts contribute.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for _, row in centerline_gdf.iterrows():
        if row['UniqueID'] and row.geometry:
            groups[row['UniqueID']].append(row.geometry)

    merged = {}
    for uid, geoms in groups.items():
        if len(geoms) == 1:
            merged[uid] = geoms[0].wkt
        else:
            # Flatten any MultiLineString into individual LineStrings
            flat = []
            for g in geoms:
                if isinstance(g, MultiLineString):
                    flat.extend(g.geoms)
                else:
                    flat.append(g)
            merged_geom = linemerge(MultiLineString(flat))
            merged[uid] = merged_geom.wkt
    return merged


def process_polygons_parallel_optimized(footprint_gdf, centerline_gdf, smooth_centerline_gdf,
                                        target_area, output_path, extension_distance, width_column, max_workers=None):
    """
    Optimized parallel processing using multiprocessing.Pool instead of ProcessPoolExecutor.
    """
    if max_workers is None:
        max_workers = min(cpu_count(), 8)

    logging.info(f"Using {max_workers} workers for parallel processing")

    # Pre-process centerlines into dictionaries using merge instead of overwrite
    centerlines_by_id = build_centerline_dict(centerline_gdf)
    smooth_centerlines_by_id = build_centerline_dict(smooth_centerline_gdf)

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

    # Process in parallel
    results = []
    with Pool(processes=max_workers) as pool:
        # Use imap for better memory efficiency and progress tracking
        for result_batch in tqdm(pool.imap(process_polygon_worker, worker_args, chunksize=10),
                                 total=len(worker_args), desc="Processing polygons"):
            if result_batch:
                results.extend(result_batch)

    # Convert results back to GeoDataFrame
    if results:
        from shapely import wkt

        # Convert WKT geometries back to shapely objects
        for result in results:
            result['geometry'] = wkt.loads(result['geometry'])

        split_polygons_gdf = gpd.GeoDataFrame(results, crs=footprint_gdf.crs)

        # Calculate real area per polygon, rounded to 1 decimal
        split_polygons_gdf['area'] = split_polygons_gdf.geometry.area.round(1)

        # Drop temporary columns
        if 'original_idx' in split_polygons_gdf.columns:
            split_polygons_gdf = split_polygons_gdf.drop(columns=['original_idx'])

        # Update output filename with real median area
        import re
        real_median = int(round(split_polygons_gdf['area'].median()))
        real_output_path = re.sub(r'_segments\d+m2\.gpkg$', f'_segments{real_median}m2.gpkg', output_path)

        # Save results
        split_polygons_gdf.to_file(real_output_path, driver="GPKG")
        logging.info(f"Split polygons saved to: {real_output_path}")
        logging.info(f"Created {len(split_polygons_gdf)} segments from {len(footprint_gdf)} polygons")
    else:
        logging.warning("No results generated from polygon splitting")


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
    footprint_path = update_path_with_suffix(cfg.dataset.ground_footprint, "_footprint_ID")
    regular_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")

    # Get the smooth centerline path if smoothing is enabled
    if cfg.split_to_plots.get("use_smooth_centerline", True) and cfg.smoothening.get("perform_smoothing", True):
        smooth_centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID_smooth")
    else:
        smooth_centerline_path = regular_centerline_path

    # Configuration parameters
    num_workers = cfg.split_to_plots.get("num_workers", None)
    segment_area = int(cfg.split_to_plots.plot_area)
    extension_distance = cfg.split_to_plots.extension_distance
    max_splitter_length = cfg.split_to_plots.max_splitter_length_buffer

    # Output path
    output_dir = os.path.dirname(footprint_path)
    output_filename = os.path.basename(footprint_path).replace(".gpkg", f"_segments{segment_area}m2.gpkg")
    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"Footprint path: {footprint_path}")
    logging.info(f"Regular centerline path: {regular_centerline_path}")
    logging.info(f"Smooth centerline path: {smooth_centerline_path}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"Parameters: segment_area={segment_area}m², extension={extension_distance}m")

    # Read input data
    try:
        logging.info("Reading footprint data...")
        footprint_gdf = read_vector_file(footprint_path)

        logging.info("Reading centerline data...")
        regular_centerline_gdf = read_vector_file(regular_centerline_path)
        smooth_centerline_gdf = read_vector_file(smooth_centerline_path)

        logging.info(f"Loaded {len(footprint_gdf)} footprints, {len(regular_centerline_gdf)} centerlines")

    except Exception as e:
        logging.error(f"Failed to read input files: {e}")
        return

    # Process polygons
    process_polygons_parallel_optimized(
        footprint_gdf,
        regular_centerline_gdf,
        smooth_centerline_gdf,
        segment_area,
        output_path,
        extension_distance=extension_distance,
        width_column=cfg.dataset.width_column,
        max_workers=num_workers
    )

    logging.info("Processing complete!")


if __name__ == "__main__":
    main()
