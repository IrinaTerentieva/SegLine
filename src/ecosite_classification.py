import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Define input and output file paths
file_path = "/media/irina/data/Blueberry/DATA/metrics/all_sites_all_metrics_PD300.gpkg"
output_path = "/media/irina/data/Blueberry/DATA/metrics/all_sites_all_metrics_PD300_ecoRF.gpkg"

# Load the dataset
gdf_original = gpd.read_file(file_path)
print('dataset length', len(gdf_original))

print(gdf_original["landcover"].unique())
# Remove rows where landcover is '0'
gdf = gdf_original[gdf_original["landcover"] != "0"]
print(gdf["landcover"].unique())

# Define feature columns
feature_cols = [
    'adjacency_density_total', 'forest_1_density', 'forest_2_density', 'forest_3_density',
    'adjacency_total_shrub_coverage', 'adjacency_tree_coverage', 'adjacency_tree13m_coverage',
    'adjacency_tree5m_coverage', 'adjacency_tree3m_coverage', 'adjacency_tree1.5m_coverage', 'hummock_coverage'
]

# Ensure all required columns exist
missing_cols = [col for col in feature_cols if col not in gdf.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

# Prepare features and labels
X = gdf[feature_cols].fillna(0)  # Fill missing values with 0
le = LabelEncoder()
y = le.fit_transform(gdf["landcover"])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

X = gdf_original[feature_cols].fillna(0)  # Fill missing values with 0

# Predict on all data
gdf_original["predicted_landcover"] = le.inverse_transform(rf_model.predict(X))
print('dataset length after preds', len(gdf_original))

# Assign ecosite based on predicted landcover
mesic_classes = ['shrubs', 'aspen', 'spruce', 'spruce and pine', 'young pine', 'young aspen']
xeric_classes = ['pine']
hydric_classes = ['tamarack fen', 'open fen', 'treed fen', 'high fen', 'low shrubs']

def assign_ecosite(landcover):
    if landcover in mesic_classes:
        return 'mesic'
    elif landcover in xeric_classes:
        return 'xeric'
    elif landcover in hydric_classes:
        return 'hydric'
    return 'unknown'

gdf_original["ecosite"] = gdf_original["predicted_landcover"].apply(assign_ecosite)

# Save the updated dataset
gdf_original.to_file(output_path, driver="GPKG")
print("Random Forest classification completed. Results saved in 'predicted_landcover' and 'ecosite' columns.")
