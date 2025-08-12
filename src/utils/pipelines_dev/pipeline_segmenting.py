#!/usr/bin/env python
"""
Debug version of split_to_plots.py that processes a single polygon
with detailed logging and visualization.
"""

import os
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, GeometryCollection, Point
from shapely.ops import split, linemerge
from shapely import wkt
import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')


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

    logging.debug(
        f"  Generating perpendiculars: avg_width={avg_width:.2f}, spacing={spacing:.2f}, centerline_length={centerline.length:.2f}")

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

    logging.debug(f"  Generated {len(perpendiculars)} perpendicular lines")
    return perpendiculars


def split_geometry(geometry, splitter):
    """
    Split a geometry with a splitter, handling GeometryCollection properly.
    """
    try:
        result = split(geometry, splitter)
        if isinstance(result, GeometryCollection):
            polygons = [geom for geom in result.geoms if isinstance(geom, Polygon)]
            # logging.debug(f"    Split result: GeometryCollection with {len(polygons)} polygons")
            return polygons
        elif isinstance(result, Polygon):
            # logging.debug(f"    Split result: Single Polygon")
            return [result]
        else:
            # logging.debug(f"    Split result: Unexpected type {type(result)}")
            return []
    except Exception as e:
        logging.debug(f"Error splitting geometry: {e}")
        return []


def visualize_splitting_process(polygon, centerline, extended_centerline, perpendiculars, segments):
    """
    Visualize the splitting process step by step.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Original polygon and centerline
    ax = axes[0, 0]
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Polygon')
    if centerline:
        if isinstance(centerline, LineString):
            x, y = centerline.xy
            ax.plot(x, y, 'r-', linewidth=1, label='Centerline')
    ax.set_title('Original Polygon and Centerline')
    ax.legend()
    ax.set_aspect('equal')

    # Plot 2: Extended centerline
    ax = axes[0, 1]
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Polygon')
    if extended_centerline and isinstance(extended_centerline, LineString):
        x, y = extended_centerline.xy
        ax.plot(x, y, 'g-', linewidth=1, label='Extended Centerline')
    ax.set_title('Extended Centerline')
    ax.legend()
    ax.set_aspect('equal')

    # Plot 3: Perpendicular lines
    ax = axes[0, 2]
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Polygon')
    for perp in perpendiculars[:10]:  # Show first 10 perpendiculars
        x, y = perp.xy
        ax.plot(x, y, 'm-', linewidth=0.5, alpha=0.5)
    ax.set_title(f'Perpendiculars ({len(perpendiculars)} total)')
    ax.set_aspect('equal')

    # Plot 4: All splitting lines
    ax = axes[1, 0]
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Polygon')
    if extended_centerline and isinstance(extended_centerline, LineString):
        x, y = extended_centerline.xy
        ax.plot(x, y, 'g-', linewidth=1, label='Extended Centerline')
    for perp in perpendiculars:
        x, y = perp.xy
        ax.plot(x, y, 'm-', linewidth=0.3, alpha=0.3)
    ax.set_title('All Splitting Lines')
    ax.legend()
    ax.set_aspect('equal')

    # Plot 5: Resulting segments
    ax = axes[1, 1]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        if isinstance(segment, Polygon):
            x, y = segment.exterior.xy
            ax.fill(x, y, color=colors[i], alpha=0.5, edgecolor='black', linewidth=0.5)
    ax.set_title(f'Result: {len(segments)} segments')
    ax.set_aspect('equal')

    # Plot 6: Segment areas histogram
    ax = axes[1, 2]
    if segments:
        areas = [seg.area for seg in segments if isinstance(seg, Polygon)]
        ax.hist(areas, bins=20, edgecolor='black')
        ax.set_xlabel('Area')
        ax.set_ylabel('Count')
        ax.set_title(f'Segment Areas (mean={np.mean(areas):.1f})')

    plt.tight_layout()
    plt.show()


def debug_single_polygon(footprint_path, centerline_path, polygon_index=0, target_area=50, extension_distance=50):
    """
    Debug the splitting process for a single polygon.
    """
    print("\n" + "=" * 60)
    print("DEBUG MODE: Processing Single Polygon")
    print("=" * 60)

    # Read data
    print(f"\n1. Reading data...")
    print(f"   Footprint: {footprint_path}")
    print(f"   Centerline: {centerline_path}")

    footprint_gdf = gpd.read_file(footprint_path)
    centerline_gdf = gpd.read_file(centerline_path)

    print(f"   Found {len(footprint_gdf)} polygons")
    print(f"   Found {len(centerline_gdf)} centerlines")

    # Select a polygon to debug
    if polygon_index >= len(footprint_gdf):
        polygon_index = 0
        print(f"   Index {polygon_index} out of range, using index 0")

    footprint_row = footprint_gdf.iloc[polygon_index]
    polygon = footprint_row.geometry
    unique_id = footprint_row.get('UniqueID', f'ID_{polygon_index}')

    print(f"\n2. Selected polygon {polygon_index}:")
    print(f"   UniqueID: {unique_id}")
    print(f"   Geometry type: {polygon.geom_type}")
    print(f"   Area: {polygon.area:.2f}")
    print(f"   Is valid: {polygon.is_valid}")

    # Find matching centerline
    centerline_match = centerline_gdf[centerline_gdf['UniqueID'] == unique_id]
    if centerline_match.empty:
        print(f"   WARNING: No centerline found for UniqueID {unique_id}")
        print(f"   Available UniqueIDs in centerlines: {centerline_gdf['UniqueID'].unique()[:5]}...")
        # Try to use first centerline as fallback
        if len(centerline_gdf) > 0:
            centerline_row = centerline_gdf.iloc[0]
            print(f"   Using first centerline as fallback")
        else:
            print("   ERROR: No centerlines available")
            return
    else:
        centerline_row = centerline_match.iloc[0]

    centerline = centerline_row.geometry
    print(f"\n3. Centerline:")
    print(f"   Geometry type: {centerline.geom_type}")
    print(f"   Length: {centerline.length:.2f}")

    # Handle MultiLineString
    if isinstance(centerline, MultiLineString):
        print(f"   MultiLineString with {len(centerline.geoms)} parts")
        centerline = linemerge(centerline)
        print(f"   After merging: {centerline.geom_type}")

    # Calculate average width
    avg_width = footprint_row.get('avg_width', polygon.area / centerline.length if centerline.length > 0 else 10)
    max_width = avg_width + 10

    if max_width <= 5:
        max_width = 15
    if avg_width >= 9:
        target_area = int(target_area * 2)

    print(f"\n4. Width calculations:")
    print(f"   Average width: {avg_width:.2f}")
    print(f"   Max width (for splitters): {max_width:.2f}")
    print(f"   Target area per segment: {target_area}")

    # Extend centerline
    print(f"\n5. Extending centerline by {extension_distance}m...")
    extended_centerline = extend_line(centerline, extension_distance)
    if isinstance(extended_centerline, LineString):
        print(f"   Extended length: {extended_centerline.length:.2f}")
        print(f"   Length increase: {extended_centerline.length - centerline.length:.2f}")

    # Generate perpendiculars
    print(f"\n6. Generating perpendicular lines...")
    perpendiculars = generate_perpendiculars(extended_centerline, avg_width, target_area, max_splitter_length=max_width)
    print(f"   Generated {len(perpendiculars)} perpendiculars")

    # Split polygon
    print(f"\n7. Splitting polygon...")

    # First split by centerline
    print(f"   Step 1: Splitting by extended centerline...")
    segments = split_geometry(polygon, extended_centerline)
    print(f"   Result: {len(segments)} segments")

    # Then split by each perpendicular
    print(f"   Step 2: Splitting by perpendiculars...")
    for i, perp in enumerate(perpendiculars):
        temp_segments = []
        for segment in segments:
            split_result = split_geometry(segment, perp)
            temp_segments.extend(split_result)
        segments = temp_segments
        if i % 10 == 0:  # Log progress every 10 perpendiculars
            print(f"     After {i + 1} perpendiculars: {len(segments)} segments")

    print(f"\n8. Final results:")
    print(f"   Total segments created: {len(segments)}")
    if segments:
        areas = [seg.area for seg in segments if isinstance(seg, Polygon)]
        print(f"   Segment areas: min={min(areas):.2f}, max={max(areas):.2f}, mean={np.mean(areas):.2f}")
        print(f"   Total area of segments: {sum(areas):.2f}")
        print(f"   Original polygon area: {polygon.area:.2f}")
        print(f"   Area difference: {abs(sum(areas) - polygon.area):.2f}")

    # Visualize
    print(f"\n9. Creating visualization...")
    visualize_splitting_process(polygon, centerline, extended_centerline, perpendiculars, segments)

    return segments


def process_all_polygons(footprint_path, centerline_path, output_path, target_area=50, extension_distance=50,
                         visualize=False):
    """
    Process all polygons and save the segmented results.
    """
    print("\n" + "=" * 60)
    print("PROCESSING ALL POLYGONS")
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

    print(f"\n2. Processing polygons...")
    for idx, footprint_row in footprint_gdf.iterrows():
        polygon = footprint_row.geometry
        unique_id = footprint_row.get('UniqueID', f'ID_{idx}')

        # Skip invalid geometries
        if not polygon or not polygon.is_valid:
            print(f"   Skipping polygon {idx} (UniqueID: {unique_id}): Invalid geometry")
            polygons_skipped += 1
            continue

        # Find matching centerline
        centerline = centerline_dict.get(unique_id)
        if not centerline:
            print(f"   Skipping polygon {idx} (UniqueID: {unique_id}): No matching centerline")
            polygons_skipped += 1
            continue

        # Handle MultiLineString
        if isinstance(centerline, MultiLineString):
            centerline = linemerge(centerline)
            if not isinstance(centerline, LineString):
                print(f"   Skipping polygon {idx} (UniqueID: {unique_id}): Could not merge MultiLineString")
                polygons_skipped += 1
                continue

        # Calculate average width
        avg_width = footprint_row.get('avg_width', polygon.area / centerline.length if centerline.length > 0 else 10)
        max_width = avg_width + 10

        if max_width <= 5:
            max_width = 15
        if avg_width >= 9:
            current_target_area = int(target_area * 2)
        else:
            current_target_area = target_area

        # Extend centerline
        extended_centerline = extend_line(centerline, extension_distance)

        # Generate perpendiculars
        perpendiculars = generate_perpendiculars(extended_centerline, avg_width, current_target_area,
                                                 max_splitter_length=max_width)

        # Split polygon
        segments = split_geometry(polygon, extended_centerline)
        for perp in perpendiculars:
            temp_segments = []
            for segment in segments:
                temp_segments.extend(split_geometry(segment, perp))
            segments = temp_segments

        # Add segments to results with metadata
        for part_id, segment in enumerate(segments):
            if isinstance(segment, Polygon) and segment.is_valid:
                segment_data = footprint_row.to_dict()
                segment_data['geometry'] = segment
                segment_data['PartID'] = part_id
                segment_data['OriginalIndex'] = idx
                segment_data['SegmentArea'] = segment.area
                all_segments.append(segment_data)

        polygons_processed += 1

        # Progress update
        if polygons_processed % 10 == 0:
            print(f"   Processed {polygons_processed}/{len(footprint_gdf)} polygons...")

    print(f"\n3. Summary:")
    print(f"   Polygons processed: {polygons_processed}")
    print(f"   Polygons skipped: {polygons_skipped}")
    print(f"   Total segments created: {len(all_segments)}")

    # Create output GeoDataFrame
    if all_segments:
        print(f"\n4. Creating output GeoDataFrame...")
        output_gdf = gpd.GeoDataFrame(all_segments, crs=footprint_gdf.crs)

        # Calculate statistics
        areas = output_gdf['SegmentArea'].values
        print(f"   Segment area statistics:")
        print(f"     Min: {areas.min():.2f} m²")
        print(f"     Max: {areas.max():.2f} m²")
        print(f"     Mean: {areas.mean():.2f} m²")
        print(f"     Median: {np.median(areas):.2f} m²")
        print(f"     Target was: {target_area} m²")

        # Save to file
        print(f"\n5. Saving to: {output_path}")
        output_gdf.to_file(output_path, driver='GPKG')
        print(f"   Saved {len(output_gdf)} segments")

        return output_gdf
    else:
        print("\n   ERROR: No segments were created!")
        return None


def main():
    """
    Main function - modify these paths to your files
    """
    # UPDATE THESE PATHS TO YOUR FILES
    footprint_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_split_polygons.gpkg"
    centerline_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_matched_centerlines.gpkg"
    output_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_segments_50m2.gpkg"

    # Parameters
    target_area = 50  # Target area for each segment in square meters
    extension_distance = 50  # How much to extend the centerline at each end

    # Option 1: Debug a single polygon (uncomment to use)
    # polygon_index = 35  # Change this to debug different polygons
    # segments = debug_single_polygon(
    #     footprint_path,
    #     centerline_path,
    #     polygon_index=polygon_index,
    #     target_area=target_area,
    #     extension_distance=extension_distance
    # )

    # Option 2: Process all polygons and save results
    output_gdf = process_all_polygons(
        footprint_path,
        centerline_path,
        output_path,
        target_area=target_area,
        extension_distance=extension_distance,
        visualize=False  # Set to True if you want to see visualizations (will be slow for many polygons)
    )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()