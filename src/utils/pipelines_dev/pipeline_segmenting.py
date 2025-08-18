#!/usr/bin/env python
"""
Fixed version that handles MultiLineStrings properly - doesn't skip polygons.
"""

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, GeometryCollection
from shapely.ops import split, linemerge
import numpy as np
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def handle_multilinestring(centerline):
    """
    Handle MultiLineString centerlines properly.
    Try to merge first, if that fails, use the longest segment.
    """
    if isinstance(centerline, LineString):
        return centerline, "LineString"

    if isinstance(centerline, MultiLineString):
        # Try to merge
        merged = linemerge(centerline)
        if isinstance(merged, LineString):
            return merged, "merged"
        else:
            # Can't merge - use longest segment
            segments = list(centerline.geoms)
            longest = max(segments, key=lambda x: x.length)
            logging.debug(f"   MultiLineString with {len(segments)} parts, using longest ({longest.length:.2f}m)")
            return longest, "longest_segment"

    return centerline, "unknown"


def extend_line(line: LineString, extension_distance=100):
    """
    Extend a line at both ends by a given distance.
    """
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


def calculate_polygon_width(polygon, centerline, num_samples=10):
    """
    Calculate the average width of a polygon.
    """
    if not isinstance(centerline, LineString):
        return polygon.area / centerline.length if hasattr(centerline, 'length') and centerline.length > 0 else 10.0

    widths = []

    for i in range(num_samples):
        distance = (i + 1) * centerline.length / (num_samples + 1)
        point = centerline.interpolate(distance)

        # Create perpendicular line
        next_point = centerline.interpolate(min(distance + 1, centerline.length))
        dx = next_point.x - point.x
        dy = next_point.y - point.y
        length = np.sqrt(dx ** 2 + dy ** 2)

        if length == 0:
            continue

        perp_x = -dy / length
        perp_y = dx / length

        max_dist = 100
        perp_line = LineString([
            (point.x - perp_x * max_dist, point.y - perp_y * max_dist),
            (point.x + perp_x * max_dist, point.y + perp_y * max_dist)
        ])

        try:
            intersection = polygon.intersection(perp_line)
            if hasattr(intersection, 'length'):
                widths.append(intersection.length)
        except:
            continue

    if widths:
        return np.mean(widths)
    else:
        return polygon.area / centerline.length if centerline.length > 0 else 10.0


def generate_perpendiculars(centerline, avg_width, target_area, max_splitter_length=10, adjustment_factor=1.35):
    """
    Generate perpendicular lines to the centerline.
    """
    if avg_width <= 0:
        avg_width = 5

    # spacing = (target_area / avg_width) * adjustment_factor
    spacing = target_area

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
    Split a geometry with a splitter.
    """
    try:
        result = split(geometry, splitter)
        if isinstance(result, GeometryCollection):
            polygons = [geom for geom in result.geoms if isinstance(geom, Polygon)]
            return polygons
        elif isinstance(result, Polygon):
            return [result]
        else:
            return []
    except Exception as e:
        return []


def process_all_polygons_fixed(footprint_path, centerline_path, output_path,
                               target_area=50, extension_distance=50, adjustment_factor=1.35):
    """
    Process all polygons with proper MultiLineString handling.
    """
    print("\n" + "=" * 60)
    print("PROCESSING ALL POLYGONS (FIXED VERSION)")
    print("=" * 60)

    # Read data
    print(f"\n1. Reading data...")
    footprint_gdf = gpd.read_file(footprint_path)
    centerline_gdf = gpd.read_file(centerline_path)

    print(f"   Found {len(footprint_gdf)} polygons")
    print(f"   Found {len(centerline_gdf)} centerlines")

    # Create a dictionary of centerlines for faster lookup
    centerline_dict = {}
    for idx, row in centerline_gdf.iterrows():
        if 'UniqueID' in row and row['UniqueID']:
            centerline_dict[row['UniqueID']] = row.geometry

    # Process each polygon
    all_segments = []
    polygons_processed = 0
    polygons_skipped = 0
    multiline_handled = 0
    handling_stats = {
        'LineString': 0,
        'merged': 0,
        'longest_segment': 0,
        'unknown': 0
    }

    print(f"\n2. Processing polygons...")
    for idx, footprint_row in footprint_gdf.iterrows():
        polygon = footprint_row.geometry
        unique_id = footprint_row.get('UniqueID', f'ID_{idx}')

        # Skip invalid geometries
        if not polygon or not polygon.is_valid:
            logging.info(f"   Skipping polygon {idx} (UniqueID: {unique_id}): Invalid geometry")
            polygons_skipped += 1
            continue

        # Find matching centerline
        centerline = centerline_dict.get(unique_id)
        if not centerline:
            logging.info(f"   Skipping polygon {idx} (UniqueID: {unique_id}): No matching centerline")
            polygons_skipped += 1
            continue

        # Handle MultiLineString properly
        original_type = centerline.geom_type
        centerline, handling_method = handle_multilinestring(centerline)
        handling_stats[handling_method] += 1

        if original_type == 'MultiLineString':
            multiline_handled += 1
            if handling_method == 'longest_segment':
                logging.debug(f"   Polygon {idx} (UniqueID: {unique_id}): Using longest segment from MultiLineString")

        # Calculate actual width
        calculated_width = calculate_polygon_width(polygon, centerline)
        avg_width = calculated_width
        max_width = min(avg_width * 1.5, avg_width + 10)

        # Adjust target area for wide polygons
        current_target_area = target_area
        if avg_width >= 15:
            current_target_area = int(target_area * 1.2)

        # Extend centerline
        extended_centerline = extend_line(centerline, extension_distance)

        # Generate perpendiculars
        perpendiculars = generate_perpendiculars(
            extended_centerline, avg_width, current_target_area,
            max_splitter_length=max_width, adjustment_factor=adjustment_factor
        )

        # Split polygon
        segments = split_geometry(polygon, extended_centerline)
        for perp in perpendiculars:
            temp_segments = []
            for segment in segments:
                temp_segments.extend(split_geometry(segment, perp))
            segments = temp_segments

        # Add segments to results with metadata
        for part_id, segment in enumerate(segments):
            if isinstance(segment, Polygon) and segment.is_valid and segment.area > 0.1:
                segment_data = footprint_row.to_dict()
                segment_data['geometry'] = segment
                segment_data['PartID'] = part_id
                segment_data['OriginalIndex'] = idx
                segment_data['RealArea'] = segment.area
                segment_data['CalculatedWidth'] = calculated_width
                segment_data['CenterlineHandling'] = handling_method
                all_segments.append(segment_data)

        polygons_processed += 1

        # Progress update
        if polygons_processed % 20 == 0:
            print(f"   Processed {polygons_processed}/{len(footprint_gdf)} polygons...")

    print(f"\n3. Summary:")
    print(f"   Polygons processed: {polygons_processed}")
    print(f"   Polygons skipped: {polygons_skipped}")
    print(f"   MultiLineStrings handled: {multiline_handled}")
    print(f"   Total segments created: {len(all_segments)}")

    print(f"\n   Centerline handling methods:")
    for method, count in handling_stats.items():
        if count > 0:
            print(f"     {method}: {count}")

    # Create output GeoDataFrame
    if all_segments:
        print(f"\n4. Creating output GeoDataFrame...")
        output_gdf = gpd.GeoDataFrame(all_segments, crs=footprint_gdf.crs)

        # Calculate statistics
        areas = output_gdf['RealArea'].values
        print(f"\n5. Segment area statistics:")
        print(f"   Min: {areas.min():.2f} m²")
        print(f"   Max: {areas.max():.2f} m²")
        print(f"   Mean: {areas.mean():.2f} m²")
        print(f"   Median: {np.median(areas):.2f} m²")
        print(f"   Target was: {target_area} m²")

        within_10 = np.sum((areas >= target_area * 0.9) & (areas <= target_area * 1.1))
        within_20 = np.sum((areas >= target_area * 0.8) & (areas <= target_area * 1.2))
        print(f"\n   Segments within target range:")
        print(f"     Within ±10% of target: {within_10} ({100 * within_10 / len(areas):.1f}%)")
        print(f"     Within ±20% of target: {within_20} ({100 * within_20 / len(areas):.1f}%)")

        # Save to file
        print(f"\n6. Saving to: {output_path}")
        output_gdf.to_file(output_path, driver='GPKG')
        print(f"   Saved {len(output_gdf)} segments")

        return output_gdf
    else:
        print("\n   ERROR: No segments were created!")
        return None


def main():
    """
    Main function - uses the SAME paths as your debug script
    """
    # These are the EXACT paths from your debug script
    spacing = 50

    # footprint_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_split_polygons.gpkg"
    # centerline_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_matched_centerlines.gpkg"
    # output_path = f"/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_{spacing}m_segments.gpkg"

    footprint_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/BRFN_SLU_pipelines_split_polygons.gpkg"
    centerline_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/BRFN_SLU_pipelines_matched_centerlines.gpkg"
    output_path = f"/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/BRFN_SLU_pipelines_{spacing}m_segments.gpkg"

    # Parameters
    target_area = 300  # Target area for each segment in square meters
    extension_distance = 50  # How much to extend the centerline at each end
    adjustment_factor = 1.35  # Adjustment to get closer to target area

    # Process all polygons with fixed MultiLineString handling
    output_gdf = process_all_polygons_fixed(
        footprint_path,
        centerline_path,
        output_path,
        target_area=spacing,
        extension_distance=extension_distance,
        adjustment_factor=adjustment_factor
    )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    # Check if oWZ9XGZr was processed
    if output_gdf is not None:
        if 'UniqueID' in output_gdf.columns:
            if 'atukXLAN' in output_gdf['UniqueID'].values:
                segments_for_id = output_gdf[output_gdf['UniqueID'] == 'atukXLAN']
                print(f"\n✅ SUCCESS: Polygon 'atukXLAN' was processed!")
                print(f"   Created {len(segments_for_id)} segments")
            else:
                print(f"\n⚠️ WARNING: Polygon 'oWZ9XGZr' was not in output!")


if __name__ == "__main__":
    main()