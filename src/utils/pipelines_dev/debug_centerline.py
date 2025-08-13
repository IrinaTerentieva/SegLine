import geopandas as gpd

def debug_matching_issue(footprint_path, centerline_path):
    """
    Debug why some polygons don't have matching centerlines.
    """
    print("\n" + "=" * 60)
    print("DEBUGGING POLYGON-CENTERLINE MATCHING")
    print("=" * 60)

    # Load data
    footprint_gdf = gpd.read_file(footprint_path)
    centerline_gdf = gpd.read_file(centerline_path)

    print(f"\nData loaded:")
    print(f"  Footprints: {len(footprint_gdf)}")
    print(f"  Centerlines: {len(centerline_gdf)}")

    # Get unique IDs from both
    footprint_ids = set(footprint_gdf['UniqueID'].values)
    centerline_ids = set(centerline_gdf['UniqueID'].values)

    print(f"\nUnique IDs:")
    print(f"  Unique footprint IDs: {len(footprint_ids)}")
    print(f"  Unique centerline IDs: {len(centerline_ids)}")

    # Find mismatches
    footprints_without_centerlines = footprint_ids - centerline_ids
    centerlines_without_footprints = centerline_ids - footprint_ids

    print(f"\nMismatches:")
    print(f"  Footprints without centerlines: {len(footprints_without_centerlines)}")
    print(f"  Centerlines without footprints: {len(centerlines_without_footprints)}")

    # Show specific missing ones
    if footprints_without_centerlines:
        print(f"\n  Missing centerlines for these footprint IDs:")
        for uid in list(footprints_without_centerlines)[:10]:  # Show first 10
            print(f"    - {uid}")
        if len(footprints_without_centerlines) > 10:
            print(f"    ... and {len(footprints_without_centerlines) - 10} more")

    # Check for the specific problematic ID
    problem_id = "Pmi76tM4"
    print(f"\nChecking specific ID: {problem_id}")

    # Check in footprints
    footprint_match = footprint_gdf[footprint_gdf['UniqueID'] == problem_id]
    if not footprint_match.empty:
        print(f"  ✓ Found in footprints (index {footprint_match.index[0]})")
    else:
        print(f"  ✗ NOT found in footprints")

    # Check in centerlines
    centerline_match = centerline_gdf[centerline_gdf['UniqueID'] == problem_id]
    if not centerline_match.empty:
        print(f"  ✓ Found in centerlines (index {centerline_match.index[0]})")
    else:
        print(f"  ✗ NOT found in centerlines")

        # Check for similar IDs (case sensitivity, spaces, etc.)
        print(f"\n  Looking for similar IDs in centerlines:")
        for cid in centerline_ids:
            if problem_id.lower() in cid.lower() or cid.lower() in problem_id.lower():
                print(f"    Similar ID found: '{cid}'")

    # Check for duplicates
    print(f"\nChecking for duplicate UniqueIDs:")
    footprint_duplicates = footprint_gdf['UniqueID'].value_counts()
    footprint_duplicates = footprint_duplicates[footprint_duplicates > 1]
    if not footprint_duplicates.empty:
        print(f"  Duplicate footprint IDs found:")
        for uid, count in footprint_duplicates.items():
            print(f"    {uid}: {count} occurrences")
    else:
        print(f"  No duplicate footprint IDs")

    centerline_duplicates = centerline_gdf['UniqueID'].value_counts()
    centerline_duplicates = centerline_duplicates[centerline_duplicates > 1]
    if not centerline_duplicates.empty:
        print(f"  Duplicate centerline IDs found:")
        for uid, count in centerline_duplicates.items():
            print(f"    {uid}: {count} occurrences")
    else:
        print(f"  No duplicate centerline IDs")

    return footprints_without_centerlines, centerlines_without_footprints


# Run this debug function
footprint_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_split_polygons.gpkg"
centerline_path = "/media/irina/My Book/Petronas/DATA/vector_data/PipelineFootprint/PipelinesOfInterest_FS_matched_centerlines.gpkg"

missing_footprints, missing_centerlines = debug_matching_issue(footprint_path, centerline_path)