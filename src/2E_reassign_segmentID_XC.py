import geopandas as gpd

input_file = r"D:\Blueberry_NEW\7_products\footprint\_combine_all\segmenting\14_trimmed_footprint_QC_plots20m2_noSlivers.gpkg"
data_full = gpd.read_file(input_file)

# Loop through each unique UniqueID
unique_ids = data_full['UniqueID'].unique()
for unique_id in unique_ids:
    data = data_full[data_full["UniqueID"] == unique_id]

    # Group by SegmentID to identify unique segment IDs
    segment_ids = data['SegmentID'].unique()

    # Iterate through each unique SegmentID
    for segment_id in segment_ids:
        features = data[data['SegmentID'] == segment_id]

        # Check if there are less than 10 features
        if len(features) < 10:

            # Select all other features that share a segment with any of these features but do not have the same SegmentID
            # related_features = data[(data.geometry.intersects(features.unary_union)) & (data['SegmentID'] != segment_id)]
            related_features = data[(data.geometry.intersects(features.geometry.union_all())) & (data['SegmentID'] != segment_id)]

            # Extract the Segment_ID from these retained features
            unique_retained_ids = related_features['SegmentID'].unique()

            if len(unique_retained_ids) == 1:
                # Reassign the Segment_ID to the current features if there is only one to choose from
                new_id = unique_retained_ids[0]
                data_full.loc[(data_full['UniqueID'] == unique_id) & (data_full['SegmentID'] == segment_id), 'SegmentID'] = new_id
            else:
                # Skip with error message
                print(f"{len(unique_retained_ids)} SegmentIDs found for SegmentID {segment_id} in UniqueID {unique_id}. Skipping. {unique_retained_ids}")

# Save the updated GeoPackage
output_path = r"D:\Blueberry_NEW\7_products\footprint\_combine_all\segmenting\15_trimmed_footprint_QC_plots20m2_noSlivers_newSegID.gpkg"
data_full.to_file(output_path, driver="GPKG")
print("Processing complete. Updated file saved.")
