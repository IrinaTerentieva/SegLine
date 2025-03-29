import geopandas as gpd
from shapely.ops import unary_union

# Processing wide and narrow lines separately bc narrow lines have 10m2 plots and wide lines have 20 m2 plots

gpkg_path = r"D:\Blueberry_NEW\7_products\footprint\_combine_all\segmenting\13_trimmed_footprint_QC_plots20m2_fixgeom.gpkg"
# layer_name = "LiDea_2024_v5_plots100m2"
gdf = gpd.read_file(gpkg_path)

# Compute the area for each polygon
gdf['area'] = gdf.geometry.area
gdf = gdf.sort_values(by='area', ascending=False)

# Define the area thresholds for different width groups
area_threshold_narrow = 5
area_threshold_wide = 10

# Create a list to track indices of polygons to remove
# indices_to_remove = []
unmatched_polys = []
remove_slivers = []

# -------------------------------------------------------------
# First, we need to get rid of slivers

# Remove slivers directly
slivers = gdf[gdf['area'] < 0.0001]
remove_slivers.extend(slivers.index.tolist())

# Remove slivers after merging (we are actually removing the larger polygon bc the larger was merged to the sliver)
gdf = gdf.drop(remove_slivers)

# -------------------------------------------------------------
# Now we can process the rest of the features, starting with narrow lines

# Do for polygons with avg_width < 9.0
for idx, row in gdf.iterrows():
    if (row['avg_width'] < 9.0) & (row['area'] < area_threshold_narrow):

        # This is the current polygon
        polygon = row.geometry
        unique_id = row['UniqueID']
        side = row['side']

        # Filter polygons sharing a line segment with the current polygon - must have same uniqueID and side
        neighbors = gdf[(gdf.geometry.touches(polygon)) & (gdf['UniqueID'] == unique_id) & (gdf['side'] == side)]

        if len(neighbors) == 1:
            # Merge the current polygon with the selected neighbor
            selected_feature = neighbors.iloc[0]
            merged_geometry = unary_union([polygon, selected_feature.geometry])

            # Update the current polygon in the GeoDataFrame
            gdf.at[idx, 'geometry'] = merged_geometry
            gdf.at[idx, 'area'] = gdf.at[idx, 'geometry'].area  # Update the area

            # Retain attributes from the selected feature
            for col in gdf.columns:
                if col not in ['geometry', 'area']:
                    gdf.at[idx, col] = selected_feature[col]

            # Drop the selected feature after merging it with the current polygon
            gdf = gdf.drop(selected_feature.name)

        elif len(neighbors) > 1:
            # Group neighbours by size
            large_neighbours = neighbors[neighbors['area'] >= area_threshold_narrow]

            if len(large_neighbours) == 1:
                selected_large = large_neighbours.iloc[0]
                geometries_to_merge = [polygon, selected_large.geometry]
                merged_geometry = unary_union(geometries_to_merge)

                # Update the current polygon in the GeoDataFrame
                gdf.at[idx, 'geometry'] = merged_geometry
                gdf.at[idx, 'area'] = gdf.at[idx, 'geometry'].area  # Update the merged polygon area

                # Retain attributes from the selected large neighbor
                for col in gdf.columns:
                    if col not in ['geometry', 'area']:
                        gdf.at[idx, col] = selected_large[col]

                # Drop the selected feature after merging with the current polygon
                gdf = gdf.drop([selected_large.name])


            elif len(large_neighbours) > 1:
                segment_id = row['SegmentID']
                matching_large_neighbors = large_neighbours[large_neighbours['SegmentID'] == segment_id]

                if len(matching_large_neighbors) == 1:
                    selected_large = matching_large_neighbors.iloc[0]
                    geometries_to_merge = [polygon, selected_large.geometry]
                    merged_geometry = unary_union(geometries_to_merge)

                    # Update the current polygon in the GeoDataFrame
                    gdf.at[idx, 'geometry'] = merged_geometry
                    gdf.at[idx, 'area'] = gdf.at[idx, 'geometry'].area  # Update the area

                    # Retain attributes from the selected large neighbor
                    for col in gdf.columns:
                        if col not in ['geometry', 'area']:
                            gdf.at[idx, col] = selected_large[col]

                    # Drop the selected feature after merging with the current polygon
                    gdf = gdf.drop([selected_large.name])


                else:
                    print(f"Skipping feature, multiple matches with different SegmentIDs: {unique_id}")
                    unmatched_polys.append(row)

# -------------------------------------------------------------
# Do the same for polygons with avg_width >= 9.0, but for different area threshold

for idx, row in gdf.iterrows():
    if (row['avg_width'] >= 9.0) & (row['area'] < area_threshold_wide):

        # This is the current polygon
        polygon = row.geometry
        unique_id = row['UniqueID']
        side = row['side']

        # Filter polygons sharing a line segment with the current polygon - must have same uniqueID and side
        neighbors = gdf[(gdf.geometry.touches(polygon)) & (gdf['UniqueID'] == unique_id) & (gdf['side'] == side)]

        if len(neighbors) == 1:
            # Merge the current polygon with the selected neighbor
            selected_feature = neighbors.iloc[0]
            merged_geometry = unary_union([polygon, selected_feature.geometry])

            # Update the current polygon in the GeoDataFrame
            gdf.at[idx, 'geometry'] = merged_geometry
            gdf.at[idx, 'area'] = gdf.at[idx, 'geometry'].area  # Update the area

            # Retain attributes from the selected feature
            for col in gdf.columns:
                if col not in ['geometry', 'area']:
                    gdf.at[idx, col] = selected_feature[col]

            # Drop the selected feature after merging it with the current polygon
            gdf = gdf.drop(selected_feature.name)

        elif len(neighbors) > 1:
            # Group neighbours by size
            large_neighbours = neighbors[neighbors['area'] >= area_threshold_wide]

            if len(large_neighbours) == 1:
                selected_large = large_neighbours.iloc[0]
                geometries_to_merge = [polygon, selected_large.geometry]
                merged_geometry = unary_union(geometries_to_merge)

                # Update the current polygon in the GeoDataFrame
                gdf.at[idx, 'geometry'] = merged_geometry
                gdf.at[idx, 'area'] = gdf.at[idx, 'geometry'].area  # Update the merged polygon area

                # Retain attributes from the selected large neighbor
                for col in gdf.columns:
                    if col not in ['geometry', 'area']:
                        gdf.at[idx, col] = selected_large[col]

                # Drop the selected feature after merging with the current polygon
                gdf = gdf.drop([selected_large.name])


            elif len(large_neighbours) > 1:
                # Select neighbor with the same SegmentID
                segment_id = row['SegmentID']
                matching_large_neighbors = large_neighbours[large_neighbours['SegmentID'] == segment_id]

                if len(matching_large_neighbors) == 1:
                    selected_large = matching_large_neighbors.iloc[0]
                    geometries_to_merge = [polygon, selected_large.geometry]
                    merged_geometry = unary_union(geometries_to_merge)

                    # Update the current polygon in the GeoDataFrame
                    gdf.at[idx, 'geometry'] = merged_geometry
                    gdf.at[idx, 'area'] = gdf.at[idx, 'geometry'].area  # Update the area

                    # Retain attributes from the selected large neighbor
                    for col in gdf.columns:
                        if col not in ['geometry', 'area']:
                            gdf.at[idx, col] = selected_large[col]

                    # Drop the selected feature after merging with the current polygon
                    gdf = gdf.drop([selected_large.name])


                else:
                    print(f"Skipping feature, multiple matches with different SegmentIDs: {unique_id}")
                    unmatched_polys.append(row)


# Convert unmatched polys to gdf
unmatched_polys_gdf = gpd.GeoDataFrame(unmatched_polys, columns=gdf.columns, crs=gdf.crs)


# Save the modified GeoDataFrame back to a new geopackage
output_path = r"D:\Blueberry_NEW\7_products\footprint\_combine_all\segmenting\14_trimmed_footprint_QC_plots20m2_noSlivers.gpkg"
output_layer_name = "noSlivers"
gdf.to_file(output_path, layer=output_layer_name, driver="GPKG")

# Output unmatched and slivers
unmatched_polys_gdf.to_file(output_path, layer="unmatched", driver="GPKG")