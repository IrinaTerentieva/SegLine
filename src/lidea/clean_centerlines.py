#!/usr/bin/env python3
"""
Clean centerline geometries: remove duplicates, subsets, and resolve overlaps.

Run after assign_id.py, before split_to_plots.py.
Operates on the _centerline_ID.gpkg file and overwrites it in place.
"""

import os
import logging
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge
import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_path_with_suffix(input_path: str, suffix: str) -> str:
    if input_path.startswith("file://"):
        input_path = input_path[7:]
    dirname = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    if filename.endswith(".shp"):
        updated_filename = filename.replace(".shp", f"{suffix}.gpkg")
    else:
        updated_filename = filename.replace(".gpkg", f"{suffix}.gpkg")
    return os.path.join(dirname, updated_filename)


def normalize_geometry(geom):
    """Merge MultiLineString into single LineString where possible."""
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
    return geom


def _get_fid(gdf, i):
    if 'fid' in gdf.columns:
        return gdf.iloc[i]['fid']
    return i


def remove_exact_duplicates(gdf, tolerance=0.01):
    """Remove lines with identical geometry (within Hausdorff tolerance)."""
    n = len(gdf)
    keep = np.ones(n, dtype=bool)
    removed = 0
    sindex = gdf.sindex

    for i in range(n):
        if not keep[i]:
            continue
        geom_i = gdf.geometry.iloc[i]
        len_i = geom_i.length

        candidates = sorted(sindex.intersection(geom_i.bounds))
        for j in candidates:
            if j <= i or not keep[j]:
                continue
            geom_j = gdf.geometry.iloc[j]
            if abs(len_i - geom_j.length) > tolerance:
                continue
            if geom_i.hausdorff_distance(geom_j) < tolerance:
                keep[j] = False
                removed += 1
                logger.info(
                    f"  Duplicate: fid={_get_fid(gdf, j)} identical to "
                    f"fid={_get_fid(gdf, i)} (length={len_i:.1f}m)"
                )

    result = gdf[keep].reset_index(drop=True)
    logger.info(f"  Removed {removed} exact duplicates. {len(result)} remaining.")
    return result


def remove_subsets(gdf, tolerance=0.5):
    """Remove lines fully contained within longer lines."""
    n = len(gdf)
    lengths = gdf.geometry.length.values
    sorted_idx = np.argsort(-lengths)

    keep = np.ones(n, dtype=bool)
    removed = 0
    sindex = gdf.sindex

    buffered_cache = {}

    for rank_i in range(n):
        i = sorted_idx[rank_i]
        if not keep[i]:
            continue
        geom_i = gdf.geometry.iloc[i]

        if i not in buffered_cache:
            buffered_cache[i] = geom_i.buffer(tolerance)
        buf_i = buffered_cache[i]

        candidates = sorted(sindex.intersection(buf_i.bounds))
        for j in candidates:
            if j == i or not keep[j]:
                continue
            if lengths[j] >= lengths[i]:
                continue
            geom_j = gdf.geometry.iloc[j]
            if buf_i.contains(geom_j):
                keep[j] = False
                removed += 1
                logger.info(
                    f"  Subset: fid={_get_fid(gdf, j)} ({lengths[j]:.1f}m) "
                    f"inside fid={_get_fid(gdf, i)} ({lengths[i]:.1f}m)"
                )

    result = gdf[keep].reset_index(drop=True)
    logger.info(f"  Removed {removed} subset lines. {len(result)} remaining.")
    return result


def _measure_overlap(geom_a, geom_b, tolerance):
    overlap = geom_b.intersection(geom_a.buffer(tolerance))
    if overlap.is_empty:
        return 0.0
    return overlap.length


def _trim_line(shorter, longer, tolerance):
    """Remove the portion of `shorter` that overlaps `longer`. Returns trimmed line or None."""
    trimmed = shorter.difference(longer.buffer(tolerance))

    if trimmed.is_empty:
        return None

    if isinstance(trimmed, MultiLineString):
        parts = sorted(trimmed.geoms, key=lambda g: g.length, reverse=True)
        trimmed = parts[0] if parts[0].length > tolerance else None
        if trimmed is None:
            return None

    if not isinstance(trimmed, LineString) or trimmed.length <= tolerance:
        return None

    # Snap the connection end back to the longer line
    start_pt = Point(trimmed.coords[0])
    end_pt = Point(trimmed.coords[-1])
    dist_start = longer.distance(start_pt)
    dist_end = longer.distance(end_pt)

    coords = list(trimmed.coords)
    if dist_start < dist_end:
        snap_pt = longer.interpolate(longer.project(start_pt))
        coords[0] = (snap_pt.x, snap_pt.y)
    else:
        snap_pt = longer.interpolate(longer.project(end_pt))
        coords[-1] = (snap_pt.x, snap_pt.y)

    return LineString(coords)


def trim_partial_overlaps(gdf, tolerance=0.5):
    """Trim shorter line where two lines partially overlap, so they share only an endpoint."""
    n = len(gdf)
    if n < 2:
        return gdf

    geometries = list(gdf.geometry)
    lengths = np.array([g.length if g is not None else 0 for g in geometries])
    sindex = gdf.sindex
    trimmed_count = 0
    removed_count = 0

    for i in range(n):
        if geometries[i] is None:
            continue
        geom_i = geometries[i]
        buf_i = geom_i.buffer(tolerance)

        candidates = sorted(sindex.intersection(buf_i.bounds))
        for j in candidates:
            if j <= i or geometries[j] is None:
                continue
            geom_j = geometries[j]

            overlap_len = _measure_overlap(geom_i, geom_j, tolerance)
            if overlap_len <= tolerance * 2:
                continue

            fid_i = _get_fid(gdf, i)
            fid_j = _get_fid(gdf, j)

            if lengths[i] >= lengths[j]:
                longer, shorter, shorter_idx = geom_i, geom_j, j
                shorter_fid, longer_fid = fid_j, fid_i
            else:
                longer, shorter, shorter_idx = geom_j, geom_i, i
                shorter_fid, longer_fid = fid_i, fid_j

            result = _trim_line(shorter, longer, tolerance)

            if result is None:
                geometries[shorter_idx] = None
                removed_count += 1
                logger.info(
                    f"  Overlap removed: fid={shorter_fid} fully within fid={longer_fid}"
                )
            else:
                geometries[shorter_idx] = result
                lengths[shorter_idx] = result.length
                trimmed_count += 1
                logger.info(
                    f"  Overlap trimmed: fid={shorter_fid} "
                    f"(overlap={overlap_len:.1f}m, remaining={result.length:.1f}m)"
                )

    if trimmed_count == 0 and removed_count == 0:
        logger.info("  No partial overlaps found.")
        return gdf

    keep_mask = [g is not None for g in geometries]
    result_gdf = gdf[keep_mask].copy()
    result_gdf['geometry'] = [g for g in geometries if g is not None]
    result_gdf = result_gdf.reset_index(drop=True)

    logger.info(
        f"  Trimmed {trimmed_count}, removed {removed_count}. {len(result_gdf)} remaining."
    )
    return result_gdf


def clean_centerlines(input_path, tolerance=0.5):
    """
    Clean centerline file in place:
      1. Remove exact geometry duplicates
      2. Remove lines fully contained within longer lines
      3. Trim partial overlaps so lines share only endpoints
    """
    gdf = gpd.read_file(input_path)
    original_count = len(gdf)
    logger.info(f"Loaded {original_count} centerlines from {input_path}")

    gdf['geometry'] = gdf.geometry.apply(normalize_geometry)
    n_multi = sum(isinstance(g, MultiLineString) for g in gdf.geometry)
    if n_multi > 0:
        logger.warning(f"  {n_multi} lines remain MultiLineString after merge")

    logger.info("Step 1: Removing exact geometry duplicates...")
    gdf = remove_exact_duplicates(gdf, tolerance=tolerance * 0.02)

    logger.info("Step 2: Removing subset lines...")
    gdf = remove_subsets(gdf, tolerance=tolerance)

    logger.info("Step 3: Resolving partial overlaps...")
    gdf = trim_partial_overlaps(gdf, tolerance=tolerance)

    gdf.to_file(input_path, driver="GPKG")
    final_count = len(gdf)
    logger.info(
        f"Done: {original_count} -> {final_count} centerlines "
        f"({original_count - final_count} removed/trimmed)"
    )
    logger.info(f"Overwritten: {input_path}")

    # Verify SegmentID survived cleaning
    if 'SegmentID' in gdf.columns:
        n_segment_ids = gdf['SegmentID'].notna().sum()
        n_unique = gdf['SegmentID'].dropna().nunique()
        logger.info(f"SegmentID verification: {n_segment_ids}/{final_count} rows have SegmentID, "
                     f"{n_unique} unique values")
    else:
        logger.warning("SegmentID column NOT found after cleaning!")

    return gdf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log_level = cfg.get("logging", {"level": "INFO"}).get("level", "INFO")
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    centerline_path = update_path_with_suffix(cfg.dataset.centerline, "_centerline_ID")
    logging.info(f"Cleaning centerline file: {centerline_path}")

    if not os.path.exists(centerline_path):
        logging.error(f"File not found: {centerline_path}")
        logging.error("Run assign_id.py first!")
        return

    clean_centerlines(centerline_path, tolerance=0.5)


if __name__ == "__main__":
    main()
