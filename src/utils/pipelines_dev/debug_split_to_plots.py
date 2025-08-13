#!/usr/bin/env python
"""
Improved debug version for splitting polygons - uses UniqueID for precise debugging.
"""

import os
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, GeometryCollection, Point
from shapely.ops import split, linemerge
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
        if not isinstance(line, LineString):
            # If can't merge, use longest segment
            line = max(line.geoms, key=lambda x: x.length)

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


def generate_perpendiculars(centerline, avg_width, target_area, max_splitter_length=10, adjustment_factor=1.35):
    """
    Generate perpendicular lines to the centerline at intervals calculated to achieve a target area.
    """
    if avg_width <= 0:
        avg_width = 5

    # Apply adjustment factor to get closer to target area
    spacing = (target_area / avg_width) * adjustment_factor

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
            return polygons
        elif isinstance(result, Polygon):
            return [result]
        else:
            return []
    except Exception as e:
        logging.debug(f"Error splitting geometry: {e}")
        return []


def calculate_polygon_width(polygon, centerline, num_samples=10):
    """
    Calculate the average width of a polygon by measuring perpendicular distances.
    """
    if not isinstance(centerline, LineString):
        return 10.0  # Default width

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

        # Extend perpendicular line to find intersection with polygon
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
        # Fallback: use area/length
        return polygon.area / centerline.length if centerline.length > 0 else 10.0


def visualize_splitting_process(polygon, centerline, extended_centerline, perpendiculars, segments, unique_id):
    """
    Visualize the splitting process step by step.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Splitting Process for Polygon: {unique_id}', fontsize=14, fontweight='bold')

    axes = []
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        axes.append(ax)

    # Plot 1: Original polygon and centerline
    ax = axes[0]
    if polygon:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-', linewidth=2, label='Polygon')
    if centerline and isinstance(centerline, LineString):
        x, y = centerline.xy
        ax.plot(x, y, 'r-', linewidth=2, label='Centerline')
    elif centerline and isinstance(centerline, MultiLineString):
        for line in centerline.geoms:
            x, y = line.xy
            ax.plot(x, y, 'r-', linewidth=2)
        ax.plot([], [], 'r-', linewidth=2, label='Centerline (Multi)')
    ax.set_title('1. Original Polygon and Centerline')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 2: Extended centerline
    ax = axes[1]
    if polygon:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-', linewidth=2, label='Polygon')
    if extended_centerline and isinstance(extended_centerline, LineString):
        x, y = extended_centerline.xy
        ax.plot(x, y, 'g-', linewidth=2, label='Extended Centerline')
    ax.set_title('2. Extended Centerline')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 3: Sample of perpendicular lines
    ax = axes[2]
    if polygon:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-', linewidth=2, label='Polygon')

    # Show subset of perpendiculars for clarity
    step = max(1, len(perpendiculars) // 20)  # Show ~20 perpendiculars
    for i, perp in enumerate(perpendiculars[::step]):
        x, y = perp.xy
        ax.plot(x, y, 'm-', linewidth=1, alpha=0.7)
    ax.set_title(f'3. Sample Perpendiculars ({len(perpendiculars)} total)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 4: All splitting lines
    ax = axes[3]
    if polygon:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-', linewidth=2, alpha=0.5, label='Polygon')
    if extended_centerline and isinstance(extended_centerline, LineString):
        x, y = extended_centerline.xy
        ax.plot(x, y, 'g-', linewidth=2, label='Centerline')
    for perp in perpendiculars:
        x, y = perp.xy
        ax.plot(x, y, 'm-', linewidth=0.3, alpha=0.3)
    ax.set_title('4. All Splitting Lines')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 5: Resulting segments
    ax = axes[4]
    if segments:
        # Use different colors for segments
        colors = plt.cm.rainbow(np.linspace(0, 1, min(len(segments), 50)))
        for i, segment in enumerate(segments[:50]):  # Show max 50 segments
            if isinstance(segment, Polygon):
                x, y = segment.exterior.xy
                color_idx = i % len(colors)
                ax.fill(x, y, color=colors[color_idx], alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.set_title(f'5. Result: {len(segments)} segments')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 6: Segment areas histogram
    ax = axes[5]
    if segments:
        areas = [seg.area for seg in segments if isinstance(seg, Polygon)]
        if areas:
            ax.hist(areas, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax.axvline(50, color='red', linestyle='--', linewidth=2, label='Target (50 m²)')
            ax.axvline(np.mean(areas), color='green', linestyle='--', linewidth=2,
                       label=f'Mean ({np.mean(areas):.1f} m²)')
            ax.set_xlabel('Area (m²)')
            ax.set_ylabel('Count')
            ax.set_title('6. Segment Area Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def debug_single_polygon(footprint_path, centerline_path, unique_id, target_area=50, extension_distance=50,
                         adjustment_factor=1.35):
    """
    Debug the splitting process for a single polygon identified by UniqueID.
    """
    print("\n" + "=" * 60)
    print(f"DEBUG MODE: Processing Polygon with UniqueID: {unique_id}")
    print("=" * 60)

    # Read data
    print(f"\n1. Reading data...")
    print(f"   Footprint: {footprint_path}")
    print(f"   Centerline: {centerline_path}")

    footprint_gdf = gpd.read_file(footprint_path)
    centerline_gdf = gpd.read_file(centerline_path)

    print(f"   Total footprints: {len(footprint_gdf)}")
    print(f"   Total centerlines: {len(centerline_gdf)}")

    # Find the specific polygon
    polygon_match = footprint_gdf[footprint_gdf['UniqueID'] == unique_id]
    if polygon_match.empty:
        print(f"\n   ERROR: No polygon found with UniqueID '{unique_id}'")
        print(f"   Available UniqueIDs (first 10): {list(footprint_gdf['UniqueID'].head(10))}")
        return None

    polygon_row = polygon_match.iloc[0]
    polygon = polygon_row.geometry

    print(f"\n2. Found polygon:")
    print(f"   UniqueID: {unique_id}")
    print(f"   Geometry type: {polygon.geom_type}")
    print(f"   Area: {polygon.area:.2f} m²")
    print(f"   Is valid: {polygon.is_valid}")

    # Find matching centerline
    centerline_match = centerline_gdf[centerline_gdf['UniqueID'] == unique_id]
    if centerline_match.empty:
        print(f"\n   ERROR: No centerline found for UniqueID '{unique_id}'")
        print(f"   This polygon has no matching centerline!")
        return None

    centerline_row = centerline_match.iloc[0]
    centerline = centerline_row.geometry

    print(f"\n3. Found centerline:")
    print(f"   Geometry type: {centerline.geom_type}")
    if isinstance(centerline, LineString):
        print(f"   Length: {centerline.length:.2f} m")
    elif isinstance(centerline, MultiLineString):
        print(f"   MultiLineString with {len(centerline.geoms)} parts")
        lengths = [line.length for line in centerline.geoms]
        print(f"   Segment lengths: {[f'{l:.2f}' for l in lengths]}")
        print(f"   Total length: {sum(lengths):.2f} m")

        # Try to merge or use longest
        merged = linemerge(centerline)
        if isinstance(merged, LineString):
            print(f"   Successfully merged into single LineString")
            centerline = merged
        else:
            print(f"   Could not merge - using longest segment")
            centerline = max(centerline.geoms, key=lambda x: x.length)
            print(f"   Selected segment length: {centerline.length:.2f} m")

    # Calculate actual width
    print(f"\n4. Width calculations:")
    calculated_width = calculate_polygon_width(polygon, centerline)
    print(f"   Calculated width: {calculated_width:.2f} m")

    # Use calculated width for splitting
    avg_width = calculated_width
    max_width = min(avg_width * 1.5, avg_width + 10)

    print(f"   Using average width: {avg_width:.2f} m")
    print(f"   Max splitter length: {max_width:.2f} m")
    print(f"   Target area per segment: {target_area} m²")
    print(f"   Adjustment factor: {adjustment_factor}")

    # Extend centerline
    print(f"\n5. Extending centerline by {extension_distance} m...")
    extended_centerline = extend_line(centerline, extension_distance)
    if isinstance(extended_centerline, LineString):
        print(f"   Extended length: {extended_centerline.length:.2f} m")
        print(f"   Length increase: {extended_centerline.length - centerline.length:.2f} m")

    # Generate perpendiculars
    print(f"\n6. Generating perpendicular lines...")
    perpendiculars = generate_perpendiculars(
        extended_centerline, avg_width, target_area,
        max_splitter_length=max_width, adjustment_factor=adjustment_factor
    )
    print(f"   Generated {len(perpendiculars)} perpendiculars")
    expected_segments = int(polygon.area / target_area)
    print(f"   Expected segments (based on area): ~{expected_segments}")

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

        if (i + 1) % 10 == 0:  # Log progress every 10 perpendiculars
            print(f"     After {i + 1} perpendiculars: {len(segments)} segments")

    print(f"\n8. Final results:")
    print(f"   Total segments created: {len(segments)}")

    if segments:
        areas = [seg.area for seg in segments if isinstance(seg, Polygon)]
        print(f"\n   Area statistics:")
        print(f"     Min area: {min(areas):.2f} m²")
        print(f"     Max area: {max(areas):.2f} m²")
        print(f"     Mean area: {np.mean(areas):.2f} m²")
        print(f"     Median area: {np.median(areas):.2f} m²")
        print(f"     Std deviation: {np.std(areas):.2f} m²")

        print(f"\n   Quality metrics:")
        print(f"     Total area of segments: {sum(areas):.2f} m²")
        print(f"     Original polygon area: {polygon.area:.2f} m²")
        print(
            f"     Area difference: {abs(sum(areas) - polygon.area):.2f} m² ({abs(sum(areas) - polygon.area) / polygon.area * 100:.1f}%)")

        within_10 = sum(1 for a in areas if target_area * 0.9 <= a <= target_area * 1.1)
        within_20 = sum(1 for a in areas if target_area * 0.8 <= a <= target_area * 1.2)
        print(f"\n   Segments close to target ({target_area} m²):")
        print(f"     Within ±10%: {within_10}/{len(areas)} ({within_10 / len(areas) * 100:.1f}%)")
        print(f"     Within ±20%: {within_20}/{len(areas)} ({within_20 / len(areas) * 100:.1f}%)")

    # Visualize
    print(f"\n9. Creating visualization...")
    visualize_splitting_process(polygon, centerline, extended_centerline, perpendiculars, segments, unique_id)

    return segments


def list_available_ids(footprint_path, centerline_path, n=20):
    """
    List available UniqueIDs for debugging.
    """
    print("\n" + "=" * 60)
    print("AVAILABLE UNIQUE IDs FOR DEBUGGING")
    print("=" * 60)

    footprint_gdf = gpd.read_file(footprint_path)
    centerline_gdf = gpd.read_file(centerline_path)

    footprint_ids = set(footprint_gdf['UniqueID'])
    centerline_ids = set(centerline_gdf['UniqueID'])
    matched_ids = footprint_ids & centerline_ids

    print(f"\nTotal footprints: {len(footprint_ids)}")
    print(f"Total centerlines: {len(centerline_ids)}")
    print(f"Matched (both files): {len(matched_ids)}")

    print(f"\nFirst {n} matched UniqueIDs you can debug:")
    for i, uid in enumerate(list(matched_ids)[:n]):
        print(f"  {i + 1}. '{uid}'")

    return list(matched_ids)


def main():
    """
    Main debug function - modify these paths to your files
    """
    # File paths
    footprint_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_split_polygons.gpkg"
    centerline_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_matched_centerlines.gpkg"

    # Option 1: List available IDs
    # available_ids = list_available_ids(footprint_path, centerline_path, n=10)

    # Option 2: Debug a specific polygon by UniqueID
    unique_id = 'atukXLAN'  # Change this to any UniqueID you want to debug

    # Parameters
    target_area = 50  # Target area for each segment in square meters
    extension_distance = 50  # How much to extend the centerline at each end
    adjustment_factor = 1.35  # Adjustment to get closer to target area

    # Run debug for specific polygon
    segments = debug_single_polygon(
        footprint_path,
        centerline_path,
        unique_id=unique_id,
        target_area=target_area,
        extension_distance=extension_distance,
        adjustment_factor=adjustment_factor
    )

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()