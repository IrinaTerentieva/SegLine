#!/usr/bin/env python
"""
Standalone script to split MultiPolygons into individual Polygons
and match them with centerlines.
"""

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import linemerge
import random
import string
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def generate_random_id(length=8):
    """Generate a random UniqueID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def split_multipolygons(footprint_path, output_path):
    """
    Split all MultiPolygons in a file into individual Polygons.
    """
    logging.info(f"\n{'=' * 60}")
    logging.info("SPLITTING MULTIPOLYGONS")
    logging.info(f"{'=' * 60}\n")

    # Read footprints
    logging.info(f"Reading footprints from: {footprint_path}")
    footprint_gdf = gpd.read_file(footprint_path)

    # Check initial geometry types
    initial_types = footprint_gdf.geometry.geom_type.value_counts()
    logging.info(f"Initial geometry types: {initial_types.to_dict()}")
    logging.info(f"Total features: {len(footprint_gdf)}")

    # Count MultiPolygons
    multi_count = initial_types.get('MultiPolygon', 0)
    if multi_count == 0:
        logging.info("No MultiPolygons found - nothing to split")
        return footprint_gdf

    logging.info(f"Found {multi_count} MultiPolygons to split")

    # Split MultiPolygons
    split_polygons = []

    for idx, row in footprint_gdf.iterrows():
        geom = row.geometry

        if geom.geom_type == 'Polygon':
            # Single polygon - keep as is
            new_row = row.to_dict()
            if 'UniqueID' not in new_row:
                new_row['UniqueID'] = generate_random_id()
            split_polygons.append(new_row)
            logging.debug(f"Row {idx}: Single Polygon - kept as is")

        elif geom.geom_type == 'MultiPolygon':
            # Split into individual polygons
            num_parts = len(geom.geoms)
            logging.info(f"Row {idx}: MultiPolygon with {num_parts} parts - splitting...")

            for i, poly in enumerate(geom.geoms):
                if poly and not poly.is_empty:
                    new_row = row.to_dict()
                    new_row['geometry'] = poly  # Replace MultiPolygon with single Polygon
                    new_row['UniqueID'] = generate_random_id()
                    new_row['mp_source'] = idx  # Track original MultiPolygon
                    new_row['mp_part'] = i  # Track which part this was
                    split_polygons.append(new_row)
                    logging.debug(f"  Created Polygon {i + 1}/{num_parts}")

    # Create new GeoDataFrame
    result_gdf = gpd.GeoDataFrame(split_polygons, crs=footprint_gdf.crs)

    # Verify results
    final_types = result_gdf.geometry.geom_type.value_counts()
    logging.info(f"\n{'=' * 40}")
    logging.info(f"RESULTS:")
    logging.info(f"  Original features: {len(footprint_gdf)}")
    logging.info(f"  Split features: {len(result_gdf)}")
    logging.info(f"  Final geometry types: {final_types.to_dict()}")

    if 'MultiPolygon' in final_types:
        logging.error("❌ ERROR: MultiPolygons still present after splitting!")
    else:
        logging.info("✅ SUCCESS: All MultiPolygons have been split into Polygons")

    # Save output
    if output_path:
        logging.info(f"\nSaving to: {output_path}")
        result_gdf.to_file(output_path, driver='GPKG')
        logging.info(f"Saved {len(result_gdf)} features")

    return result_gdf


def match_centerlines_to_polygons(polygon_gdf, centerline_path, output_centerline_path):
    """
    Match centerlines to polygons based on spatial overlap.
    """
    logging.info(f"\n{'=' * 60}")
    logging.info("MATCHING CENTERLINES TO POLYGONS")
    logging.info(f"{'=' * 60}\n")

    # Read centerlines
    logging.info(f"Reading centerlines from: {centerline_path}")
    centerline_gdf = gpd.read_file(centerline_path)

    centerline_types = centerline_gdf.geometry.geom_type.value_counts()
    logging.info(f"Centerline types: {centerline_types.to_dict()}")
    logging.info(f"Total centerlines: {len(centerline_gdf)}")

    # Split MultiLineStrings if present
    if 'MultiLineString' in centerline_types:
        logging.info("Splitting MultiLineStrings...")
        split_lines = []

        for idx, row in centerline_gdf.iterrows():
            geom = row.geometry

            if geom.geom_type == 'LineString':
                split_lines.append(row.to_dict())
            elif geom.geom_type == 'MultiLineString':
                for i, line in enumerate(geom.geoms):
                    if line and not line.is_empty:
                        new_row = row.to_dict()
                        new_row['geometry'] = line
                        split_lines.append(new_row)

        lines_gdf = gpd.GeoDataFrame(split_lines, crs=centerline_gdf.crs)
        logging.info(f"Split into {len(lines_gdf)} line segments")
    else:
        lines_gdf = centerline_gdf

    # Create spatial index
    logging.info("Building spatial index...")
    lines_sindex = lines_gdf.sindex

    # Match lines to polygons
    matched_centerlines = []

    for poly_idx, poly_row in polygon_gdf.iterrows():
        polygon = poly_row.geometry
        unique_id = poly_row.get('UniqueID', generate_random_id())

        # Find overlapping lines
        bounds = polygon.bounds
        possible_matches = list(lines_sindex.intersection(bounds))

        overlapping_lines = []
        for line_idx in possible_matches:
            line = lines_gdf.iloc[line_idx].geometry
            if polygon.intersects(line):
                # Check if significant portion is within polygon
                try:
                    intersection = polygon.intersection(line)
                    if intersection.length / line.length > 0.3:  # 30% threshold
                        overlapping_lines.append(line)
                except:
                    pass

        if overlapping_lines:
            # Merge lines for this polygon
            if len(overlapping_lines) == 1:
                merged = overlapping_lines[0]
            else:
                merged = linemerge(overlapping_lines)

            centerline_row = {
                'UniqueID': unique_id,
                'geometry': merged,
                'num_segments': len(overlapping_lines)
            }
            matched_centerlines.append(centerline_row)
            logging.debug(f"Polygon {poly_idx}: matched {len(overlapping_lines)} lines")

    # Create result
    if matched_centerlines:
        result_centerlines = gpd.GeoDataFrame(matched_centerlines, crs=centerline_gdf.crs)
        logging.info(f"\nMatched {len(result_centerlines)} centerlines to polygons")

        if output_centerline_path:
            result_centerlines.to_file(output_centerline_path, driver='GPKG')
            logging.info(f"Saved centerlines to: {output_centerline_path}")

        return result_centerlines
    else:
        logging.warning("No centerlines matched to polygons")
        return None


def main():
    """
    Main function to split MultiPolygons and match centerlines.
    """
    # Input paths - UPDATE THESE
    footprint_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS.shp"
    centerline_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_centerlines_buf5_rowbyrow_str.gpkg"

    # Output paths
    output_footprint = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_split_polygons.gpkg"
    output_centerline = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_matched_centerlines.gpkg"

    print("\n" + "=" * 60)
    print("MULTIPOLYGON SPLITTER FOR PIPELINES")
    print("=" * 60)

    # Step 1: Split MultiPolygons
    polygon_gdf = split_multipolygons(footprint_path, output_footprint)

    # Step 2: Match centerlines to split polygons
    if polygon_gdf is not None and len(polygon_gdf) > 0:
        centerlines_gdf = match_centerlines_to_polygons(
            polygon_gdf, centerline_path, output_centerline
        )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Split polygons: {output_footprint}")
    print(f"  Matched centerlines: {output_centerline}")

    # Final verification
    if output_footprint:
        verify_gdf = gpd.read_file(output_footprint)
        types = verify_gdf.geometry.geom_type.value_counts()
        print(f"\nFinal verification of {output_footprint}:")
        print(f"  Geometry types: {types.to_dict()}")
        if 'MultiPolygon' in types:
            print("  ⚠️ WARNING: MultiPolygons still present!")
        else:
            print("  ✅ All features are single Polygons")


if __name__ == "__main__":
    main()