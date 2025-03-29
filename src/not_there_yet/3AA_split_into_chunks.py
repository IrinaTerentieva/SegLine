import os
import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree
import pandas as pd

import os
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

def split_segment_gpkg(segment_gpkg, seedling_gpkg, output_dir, num_splits=4, buffer_distance=30):
    """
    Splits a segment GeoPackage into `num_splits` smaller files, ensuring segments and their sub-plots stay together,
    and assigns seedlings within a `buffer_distance` of each segment.

    Parameters:
    - segment_gpkg (str): Path to the segment GeoPackage.
    - seedling_gpkg (str): Path to the corresponding seedling GeoPackage.
    - output_dir (str): Directory to save the split files.
    - num_splits (int): Number of parts to split into.
    - buffer_distance (float): Buffer distance (meters) around segments to retain seedlings.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the segment and seedling data
    print(f"📂 Loading segments: {segment_gpkg}")
    segments_gdf = gpd.read_file(segment_gpkg)
    print(f"📂 Loading seedlings: {seedling_gpkg}")
    seedlings_gdf = gpd.read_file(seedling_gpkg)

    # Ensure both datasets have the same CRS
    if seedlings_gdf.crs != segments_gdf.crs:
        seedlings_gdf = seedlings_gdf.to_crs(segments_gdf.crs)

    # **Step 1: Group by UniqueID and SegmentID to keep segments and sub-plots together**
    segment_groups = [group for _, group in segments_gdf.groupby(["UniqueID", "SegmentID"])]

    # Ensure num_splits does not exceed the number of available segment groups
    num_chunks = min(num_splits, len(segment_groups))

    # Manually split the list into `num_chunks` while keeping segments and their sub-plots together
    chunk_size = max(1, len(segment_groups) // num_chunks)  # Ensure at least 1 segment per chunk
    segment_chunks = [segment_groups[i:i + chunk_size] for i in range(0, len(segment_groups), chunk_size)]

    print(f"🔹 Splitting {len(segment_groups)} segment groups into {num_chunks} chunks...")

    # **Step 2: Assign seedlings within `buffer_distance` of each segment chunk**
    seedlings_tree = STRtree(seedlings_gdf.geometry)  # Build spatial index for fast queries

    for i, segment_chunk in enumerate(segment_chunks):
        if len(segment_chunk) == 0:
            continue  # Skip empty chunks

        # Merge segment groups into a single GeoDataFrame **without dissolving sub-plots**
        segment_chunk_gdf = gpd.GeoDataFrame(pd.concat(segment_chunk, ignore_index=True), crs=segments_gdf.crs)

        # Output file paths
        segment_output_path = os.path.join(output_dir, f"segment_chunk_{i+1}.gpkg")
        seedling_output_path = os.path.join(output_dir, f"seedling_chunk_{i+1}.gpkg")

        print(f"🟢 Processing chunk {i+1}...")

        # **Step 3: Create a buffer around each individual segment (NOT dissolved)**
        segment_buffer = segment_chunk_gdf.copy()
        segment_buffer["geometry"] = segment_buffer.geometry.buffer(buffer_distance)  # Keep each segment separate

        # Find seedlings within the buffered area **without dissolving it**
        possible_seedlings_idx = seedlings_tree.query(segment_buffer.geometry.unary_union)
        matching_seedlings = seedlings_gdf.iloc[possible_seedlings_idx]

        # Ensure seedlings are truly within the buffered region (not just bounding box)
        matching_seedlings = matching_seedlings[matching_seedlings.geometry.intersects(segment_buffer.geometry.unary_union)]

        # **Step 4: Save the split segments and seedlings**
        segment_chunk_gdf.to_file(segment_output_path, driver="GPKG")
        matching_seedlings.to_file(seedling_output_path, driver="GPKG")

        print(f"✅ Saved {len(segment_chunk_gdf)} segments (with sub-plots) to {segment_output_path}")
        print(f"✅ Saved {len(matching_seedlings)} seedlings to {seedling_output_path}")

    print("🚀 Splitting complete!")


# **Batch Process Multiple Sites**
def process_all_sites(segment_dir, seedling_dir, output_dir, num_splits=4, buffer_distance=30):
    """
    Processes all segment and seedling GeoPackages in the given directories.
    Matches files by the first 9 characters of their filenames.

    Parameters:
    - segment_dir (str): Directory containing segment GeoPackages.
    - seedling_dir (str): Directory containing seedling GeoPackages.
    - output_dir (str): Directory to save the output split files.
    - num_splits (int): Number of pieces to split each segment GPKG into.
    - buffer_distance (float): Buffer distance (meters) around segments to retain seedlings.
    """

    # Get all segment files
    segment_files = [f for f in os.listdir(segment_dir) if f.endswith(".gpkg")]

    # Create a dictionary of seedling files, indexed by their first 9 characters
    seedling_files = {f[:20]: os.path.join(seedling_dir, f) for f in os.listdir(seedling_dir) if f.endswith(".gpkg")}

    for segment_file in segment_files:
        segment_path = os.path.join(segment_dir, segment_file)

        # Extract the first 9 characters of the segment filename
        segment_prefix = segment_file[:20]

        # Try to find a matching seedling file
        matched_seedling_file = seedling_files.get(segment_prefix)

        if not matched_seedling_file:
            print(f"⚠ Warning: No seedling file found for {segment_file} (Prefix: {segment_prefix}). Skipping...")
            continue

        print(f"✅ Matched {segment_file} with {os.path.basename(matched_seedling_file)}")

        # Create an output subdirectory for this site
        site_output_dir = os.path.join(output_dir, segment_prefix)
        os.makedirs(site_output_dir, exist_ok=True)

        # Process this site
        split_segment_gpkg(segment_path, matched_seedling_file, site_output_dir, num_splits=num_splits, buffer_distance=buffer_distance)

    print("✅ All sites processed successfully!")

# **Run the Script for All Sites**
if __name__ == "__main__":
    # Define input and output directories
    segment_dir = r"D:\Blueberry_NEW\7_products\footprint\_combine_all\segmenting\split_by_site\chunking"
    seedling_dir = r"D:\Blueberry_NEW\7_products\PD300\seedlings\chunking"
    output_dir = r"D:\Blueberry_NEW\7_products\PD300\seedlings\chunking\chunked"

    num_splits = 18  # Change this to adjust the number of splits
    buffer_distance = 30  # Buffer distance in meters

    process_all_sites(segment_dir, seedling_dir, output_dir, num_splits, buffer_distance)
