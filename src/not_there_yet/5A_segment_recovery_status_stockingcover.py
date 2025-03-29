import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_segment_length(plots_gdf):
    """
    Computes length of each segment using 'avg_width' and 'segment_area'
    """
    # Compute segment_length using avg_width and segment_area
    plots_gdf['segment_length'] = plots_gdf['segment_area'] / plots_gdf['avg_width']

    return plots_gdf


def switch_regions(plots_gdf):
    """
    Update regions - lidea II and lidea south labels were switched somewhere in the attribution - switch them back
    """

    # Dictionary for renaming regions to site names
    region_to_site_mapping = {
        "GrizzlyCreekSite": "Grizzly Creek",
        "PA1-PinkMtnRanchRestored": "Pink Mountain",
        "PA2-E1(East)-AtticCreekRoadSite": "Attic Creek",
        "PA2-E2(East)-BeattonRiver-SiteA": "Beatton River",
        "PA2-E2(East)-BeattonRiver-SiteB": "Dog Rib",
        "PA2-W2(West)-RestoredWellpadAccess": "Wellpad Access",
        "PA2-W2(West)-SouthSikanniRoad": "South Sikanni"
    }

    # Apply the mapping
    plots_gdf["sitename"] = plots_gdf["sitename"].replace(region_to_site_mapping)

    return plots_gdf

def update_lowdensitywetland(plots_gdf):
    # Update dominant_landcover for low density treed fens based on low adjacency coverage
    low_density_mask = ((plots_gdf['ecosite'] == 'hydric') & (plots_gdf['adjacency_tree1.5m_coverage'] < 25))
    plots_gdf.loc[low_density_mask, 'predicted_landcover'] = 'open fen'

    return plots_gdf


def compute_stocking(plots_gdf, cover_threshold = 50, count='total_count'):

    plots_gdf['segment_stocking'] = 0.0
    plots_gdf['segment_stocking_method'] = 'none'
    plots_gdf['segment_coverage_side0'] = 0.0
    plots_gdf['segment_coverage_side1'] = 0.0
    plots_gdf['plot_status'] = 'not stocked'
    plots_gdf['plot_stocking_method'] = 'none'

    # Group polygons by UniqueID first
    unique_id_groups = plots_gdf.groupby("UniqueID")

    # Count the total number of segments for the progress bar
    total_segments = sum(len(unique_group.groupby("SegmentID")) for _, unique_group in unique_id_groups)

    # Create a single progress bar for all segments
    with tqdm(total=total_segments, desc="Computing stocking") as pbar:

        for unique_id, unique_group in unique_id_groups:

            # if unique_id != 'YZkeW10S':
            #     continue

            # Now group by SegmentID within each UniqueID
            segment_groups = unique_group.groupby("SegmentID")

            for segment_id, segment_group in segment_groups:

                # if segment_id != 11:
                #     continue

                # print('Len of segment group 11: ', len(segment_group))

                # Skip stocking calculation for segments in Aspen forests with lots of overhang - we will assess these later based on coverage
                if (
                    ((segment_group["predicted_landcover"] == "aspen").any() or
                     (segment_group["adjacency_tree13m_coverage"] >= 80).any()) and
                    (segment_group["segment_forest_%cover"] >= 30).any()
                ):
                    # print('Aspen!')
                    continue

                # Total number of plots in the segment
                total_plots = segment_group["plot_id"].nunique()
                # print('Total plots: ', total_plots)

                # Variables to store side stocking counts
                side_stocking_counts = {0: 0, 1: 0}

                for side in [0, 1]:
                    # print('Side: ', side)
                    # Filter plots belonging to the current side
                    side_plots = segment_group[segment_group["side"] == side]

                    # Loop through each plot in the side and determine if it is stocked
                    for idx, plot in side_plots.iterrows():
                        # print(plot.plot_id)
                        # For aspen forests, assess cover in the 3-5 m range. For every other ecosite, assess cover in 3+ m range
                        if plot['predicted_landcover'] == 'aspen' or plot['adjacency_tree13m_coverage'] >= 80:
                            cover_value = plot['plot_tall_veg_%cover']
                        else:
                            cover_value = plot['plot_tall_veg_%cover'] + plot['plot_forest_%cover']

                        # Assess if each plot is stocked either by percent cover or by seedling detections
                        if cover_value >= cover_threshold:
                            # Mark plot as stocked by cover
                            plots_gdf.at[idx, 'plot_status'] = 'stocked'
                            plots_gdf.at[idx, 'plot_stocking_method'] = 'cover'
                        elif plot['avg_width'] < 9 and plot[count] >= 1:    # Consider 10m2 plots wth one seedling as 'stocked'
                            # Mark plot as stocked by detection
                            plots_gdf.at[idx, 'plot_status'] = 'stocked'
                            plots_gdf.at[idx, 'plot_stocking_method'] = 'detection'
                        elif plot['avg_width'] >= 9 and plot[count] >= 2:   # Consider 20m2 plots wth two seedlings as 'stocked'
                            # Mark plot as stocked by detection
                            plots_gdf.at[idx, 'plot_status'] = 'stocked'
                            plots_gdf.at[idx, 'plot_stocking_method'] = 'detection'
                        elif plot['avg_width'] >= 9 and plot[count] == 1:   # Consider 20m2 plots wth one seedling as 'half stocked'
                            # Mark plot as stocked by detection
                            plots_gdf.at[idx, 'plot_status'] = 'half stocked'
                            plots_gdf.at[idx, 'plot_stocking_method'] = 'detection'

                    # Refresh side plots and segment group to reflect the updates to plots_gdf
                    side_plots = plots_gdf.loc[side_plots.index]
                    # print('Number of plots: ', len(side_plots))
                    # print('index: ', side_plots[['UniqueID', 'SegmentID', 'plot_id']])

                    # Count stocked plots in the side
                    side_plots_stocked = side_plots[side_plots['plot_status'] == 'stocked']["plot_id"].nunique()
                    # print('side_plots_stocked: ', side_plots_stocked)

                    # Count half-stocked plots in the side (this only will apply for wide lines) - divide by two (bc they only count as half stocked)
                    side_plots_halfstocked = (side_plots[side_plots['plot_status'] == 'half stocked']["plot_id"].nunique()) / 2
                    # print('side_plots_halfstocked: ', side_plots_halfstocked)

                    # Compute stocking for current side
                    side_stocking = ((side_plots_stocked + side_plots_halfstocked) / len(side_plots)) * 100 if len(side_plots) > 0 else 0
                    # print(side_stocking)

                    # Assign the side stocking percentage to all rows in the segment
                    plots_gdf.loc[segment_group.index, f'segment_coverage_side{side}'] = side_stocking

                    # Update side stocking counts for total stocking calculation (include full and half stocked plots)
                    side_stocking_counts[side] = (side_plots_stocked + side_plots_halfstocked)

                # Calculate total stocking for the segment
                total_stocking = (side_stocking_counts[0] + side_stocking_counts[1]) / total_plots * 100 if total_plots > 0 else 0

                # Assign total stocking to all rows in the segment group
                plots_gdf.loc[segment_group.index, 'segment_stocking'] = total_stocking

                # Refresh segment_group after updating the plots_gdf
                segment_group = plots_gdf.loc[segment_group.index]

                # Determine segment_stocking_method based on plot_stocking_method in the segment group
                stocking_method = segment_group['plot_stocking_method'].unique()
                if 'cover' in stocking_method and 'detection' in stocking_method:
                    plots_gdf.loc[segment_group.index, 'segment_stocking_method'] = 'detection+cover'
                elif 'cover' in stocking_method:
                    plots_gdf.loc[segment_group.index, 'segment_stocking_method'] = 'cover'
                elif 'detection' in stocking_method:
                    plots_gdf.loc[segment_group.index, 'segment_stocking_method'] = 'detection'

                # Update the progress bar after processing the segment
                pbar.update(1)

    return plots_gdf


def assess_stocking(plots_gdf):
    """
    Determine if stocking passes or fails "framework" criteria, depending on updated ecosite
    """
    # Initialize columns to track stocking passes
    plots_gdf['_stocked'] = 0

    # Assess stocking in all sites, except aspen stands with lots of overhang
    plots_gdf.loc[(
            ((plots_gdf['ecosite'].isin(['xeric'])) & (plots_gdf['segment_stocking'] >= 50)) |    # Stocking threshold for xeric
            (((plots_gdf['ecosite'].isin(['hydric'])) & (plots_gdf['adjacency_tree1.5m_coverage'] < 25)) & (plots_gdf['segment_stocking'] >= 50)) |     # Stocking threshold for low density treed wetlands
            (((plots_gdf['ecosite'].isin(['hydric'])) & (plots_gdf['adjacency_tree1.5m_coverage'] >= 25)) & (plots_gdf['segment_stocking'] >= 70)) |    # Stocking threshold for treed wetlands
            (((plots_gdf['ecosite'].isin(['mesic'])) & ~((plots_gdf['predicted_landcover'].isin(['aspen'])) | (plots_gdf['adjacency_tree13m_coverage'] >= 80))) & (plots_gdf['segment_stocking'] >= 70)) |    # Stocking threshold for mesic stands that are not aspen forests
            ((((plots_gdf['predicted_landcover'].isin(['aspen'])) | (plots_gdf['adjacency_tree13m_coverage'] >= 80)) & (plots_gdf['segment_forest_%cover'] < 30)) & (plots_gdf['segment_stocking'] >= 70)) |  # Stocking threshold for mesic (with low overhang)

            # Stocking for mesic (with lots of overhang)
            ((((plots_gdf['predicted_landcover'].isin(['aspen'])) | (plots_gdf['adjacency_tree13m_coverage'] >= 80)) &
              (plots_gdf['segment_forest_%cover'] >= 30)) &
             (((plots_gdf['segment_tall_veg_%cover'] / (100 - plots_gdf['segment_forest_%cover'])) * 100) >= 50))
    ), '_stocked'] = 1

    return plots_gdf


def assess_coverage(plots_gdf):
    """
    Determine if coverage passes or fails "framework" criteria, depending on updated ecosite
    """
    plots_gdf['_balanced'] = 0

    # Assess coverage/balance in all sites, except aspen stands with lots of overhang
    plots_gdf.loc[(
            ((plots_gdf['ecosite'].isin(['xeric'])) &
             ((plots_gdf['segment_coverage_side0'] >= 40) & (plots_gdf['segment_coverage_side1'] >= 40))) |  # Coverage threshold for xeric

            (((plots_gdf['ecosite'].isin(['hydric'])) & (plots_gdf['adjacency_tree1.5m_coverage'] < 25)) &
             ((plots_gdf['segment_coverage_side0'] >= 40) & (plots_gdf['segment_coverage_side1'] >= 40))) |  # Coverage threshold for low density treed wetlands

            (((plots_gdf['ecosite'].isin(['hydric'])) & (plots_gdf['adjacency_tree1.5m_coverage'] >= 25)) &
             ((plots_gdf['segment_coverage_side0'] >= 50) & (plots_gdf['segment_coverage_side1'] >= 50))) | # Coverage threshold for treed wetlands

            (((plots_gdf['ecosite'].isin(['mesic'])) & ~((plots_gdf['predicted_landcover'].isin(['aspen'])) | (plots_gdf['adjacency_tree13m_coverage'] >= 80))) &
             ((plots_gdf['segment_coverage_side0'] >= 50) & (plots_gdf['segment_coverage_side1'] >= 50))) |    # Coverage threshold for mesic stands that are not aspen forests

            ((((plots_gdf['predicted_landcover'].isin(['aspen'])) | (plots_gdf['adjacency_tree13m_coverage'] >= 80)) &
              (plots_gdf['segment_forest_%cover'] < 30)) &
             ((plots_gdf['segment_coverage_side0'] >= 50) & (plots_gdf['segment_coverage_side1'] >= 50)))   # Coverage threshold for mesic (with low overhang)
    ), '_balanced'] = 1

    return plots_gdf


def assess_shrubs(plots_gdf):
    """
    Determine if there is a high coverage of shrubs 1-3 m tall
    """
    # If cover of 1-3 m is above 50%, assign _shrubs to 1
    plots_gdf['_shrubs'] = (plots_gdf['segment_medium_veg_%cover'] >= 50).astype(int)

    plots_gdf['_shrubs'] = 0

    # Assess stocking in all sites, except aspen stands with lots of overhang
    plots_gdf.loc[(
            ((plots_gdf['ecosite'].isin(['xeric', 'hydric'])) & (
                (plots_gdf['segment_medium_veg_%cover'] + plots_gdf['segment_tall_veg_%cover'] + plots_gdf['segment_forest_%cover']) >= 50)) |  # Shrubby for xeric and hydric (1+ m coverage >50%)

            (((plots_gdf['ecosite'].isin(['mesic'])) & ~(
                (plots_gdf['predicted_landcover'].isin(['aspen'])) | (plots_gdf['adjacency_tree13m_coverage'] >= 80))) & (
                    (plots_gdf['segment_medium_veg_%cover'] + plots_gdf['segment_tall_veg_%cover'] + plots_gdf['segment_forest_%cover']) >= 50)) |  # Shrubby for mesic stands that are not aspen forests (1+ m coverage >50%)

            (((plots_gdf['predicted_landcover'].isin(['aspen'])) | (plots_gdf['adjacency_tree13m_coverage'] >= 80)) & (
                    (plots_gdf['segment_medium_veg_%cover'] + plots_gdf['segment_tall_veg_%cover']) >= 50))   # Shrubby for mesic stands that ARE aspen forests (1-5 m coverage >50%)
    ), '_shrubs'] = 1
    return plots_gdf


def assess_overhang(plots_gdf):
    """
    Attribute lines in aspen stands with overhang > 30%
    """
    plots_gdf['_overhang'] = (
        # Assign overhang tag to segments with lots of overhang in aspen stands
        (((plots_gdf['predicted_landcover'].isin(['aspen'])) |
           (plots_gdf['adjacency_tree13m_coverage'] >= 80)) &
          (plots_gdf['segment_forest_%cover'] >= 30))
    ).astype(int)

    return plots_gdf


def assess_shadows(plots_gdf):
    """
    Assess shadows per segment
    """
    # # Get shadow info from the shadows_gdf
    # plots_gdf = plots_gdf.merge(
    #     shadows_gdf[['UniqueID', 'SegmentID', 'shadow_classification']],  # We need shadow_classification
    #     on=['UniqueID', 'SegmentID'],  # Join using these fields
    #     how='left'  # Keep all rows from plots_gdf
    # )
    plots_gdf['_dark_imagery'] = (plots_gdf['shadow_coverage'] > 75).astype(int)

    # plots_gdf = plots_gdf.drop(columns=['shadow_classification'])

    return plots_gdf


def assess_trails(plots_gdf):

    # Group polygons by UniqueID first
    unique_id_groups = plots_gdf.groupby("UniqueID")

    # Count the total number of segments for the progress bar
    total_segments = sum(len(unique_group.groupby("SegmentID")) for _, unique_group in unique_id_groups)

    # Create a single progress bar for all segments
    with tqdm(total=total_segments, desc="Processing trails") as pbar:
        # Initialize new column for trails
        plots_gdf['_trails'] = 0

        for unique_id, unique_group in unique_id_groups:

            # Now group by SegmentID within each UniqueID
            segment_groups = unique_group.groupby("SegmentID")

            for segment_id, segment_group in segment_groups:
                # Calculate the total number of plots in the segment
                total_plots = len(segment_group)

                # Calculate the percentage of plots meeting each condition
                condition_1 = (segment_group['plot_trail_coverage'] >= 20).sum() / total_plots
                condition_2 = (segment_group['plot_trail_coverage'] >= 10).sum() / total_plots

                # Check if either condition is satisfied
                if condition_1 >= 0.4 or condition_2 >= 0.6:
                    plots_gdf.loc[segment_group.index, '_trails'] = 2   # Heavy trails
                elif 0.6 > condition_2 >= 0.4:
                    plots_gdf.loc[segment_group.index, '_trails'] = 1  # Medium trails

                # Update progress bar
                pbar.update(1)

    return plots_gdf


def assign_untreated(plots_gdf):
    """
    Assign NULL treatments as 'Untreated' bc they are not in treatment layer
    """

    plots_gdf['dominant_treatment'] = 'Untreated'
    # plots_gdf['dominant_treatment'] = plots_gdf['dominant_treatment'].fillna('Untreated')

    return plots_gdf


def recovery_status(plots_gdf):
    """
    Determine final recovery status based on stocking and coverage
    """
    conditions = [(plots_gdf['_stocked'] == 1) & (plots_gdf['_balanced'] == 1),
                  (plots_gdf['_stocked'] == 1) & (plots_gdf['_balanced'] == 0),
                  (plots_gdf['_stocked'] == 0)]

    choices = ['SR', 'CSR', 'NSR']
    # If any condition is met the coverage will "Pass", otherwise it will "Fail"
    plots_gdf['segment_recovery_status'] = np.select(conditions, choices, default='')

    return plots_gdf


def assess_all_plots(plots_gpkg, seedlings='total_count'):

    plots_gdf = gpd.read_file(plots_gpkg)

    # Compute length of segments
    plots_gdf = compute_segment_length(plots_gdf)

    # Switch project regions because they're wrong
    plots_gdf = switch_regions(plots_gdf)

    # # Update ecosites
    # plots_gdf = update_lowdensitywetland(plots_gdf)

    # Compute stocking for all plots
    plots_gdf = compute_stocking(plots_gdf, count=seedlings)

    # Assess stocking
    plots_gdf = assess_stocking(plots_gdf)

    # Assess coverage
    plots_gdf = assess_coverage(plots_gdf)

    # Assess shrubs
    plots_gdf = assess_shrubs(plots_gdf)

    # Assess overhang
    plots_gdf = assess_overhang(plots_gdf)

    # Assess shadows
    plots_gdf = assess_shadows(plots_gdf)

    # Assess trails - could probably use some work (e.g., for a trail stocking or something)
    plots_gdf = assess_trails(plots_gdf)

    # Assess final recovery status
    plots_gdf = recovery_status(plots_gdf)
    # if plots_gdf['segment_recovery_status'] == 'SR':
    #     print('Hoooray!')

    # Assign NULL treatment values to 'Untreated'
    plots_gdf = assign_untreated(plots_gdf)

    output_path = plots_gpkg.replace('_v4.2.gpkg', '_v4.2_ass_allseedlings.gpkg')
    plots_gdf.to_file(output_path, driver="GPKG")


if __name__ == "__main__":
    # Input paths
    for PD in ['PD300', 'PD25', 'PD5']:   #
        plots_gpkg = f"/media/irina/data/Blueberry/DATA/metrics/all_sites_all_metrics_{PD}_v4.2.gpkg"

        seedlings = 'plot_big_inner_count'
        seedlings = 'plot_total_inner_count'

        assess_all_plots(plots_gpkg, seedlings = seedlings)

