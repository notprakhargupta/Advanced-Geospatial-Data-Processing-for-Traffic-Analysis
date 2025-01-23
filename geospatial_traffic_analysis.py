
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import folium

# Step 1: Load Geospatial Data
def load_data(file_path):
    """Load geospatial traffic data from a file."""
    data = gpd.read_file(file_path)
    print("Data loaded successfully.")
    return data

# Step 2: Data Preprocessing
def preprocess_data(data):
    """Preprocess the data for modeling."""
    data = data.dropna()
    data['congestion_level'] = data['traffic_density'].apply(lambda x: 1 if x > 50 else 0) # Example threshold
    print("Data preprocessing completed.")
    return data

# Step 3: Feature Engineering
def feature_engineering(data):
    """Create features from geospatial data."""
    data['latitude'] = data.geometry.y
    data['longitude'] = data.geometry.x
    features = data[['latitude', 'longitude', 'time_of_day']]
    labels = data['congestion_level']
    print("Feature engineering completed.")
    return features, labels

# Step 4: Train Machine Learning Model
def train_model(features, labels):
    """Train a machine learning model to predict congestion."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Accuracy: {accuracy:.2f}")
    return model

# Step 5: Visualize Traffic Data on Map
def visualize_data(data, map_file):
    """Visualize traffic data on a Folium map."""
    m = folium.Map(location=[data.geometry.y.mean(), data.geometry.x.mean()], zoom_start=12)
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color='red' if row['congestion_level'] == 1 else 'green',
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)
    m.save(map_file)
    print(f"Traffic data visualized. Map saved to {map_file}")

# Main Execution
def main():
    file_path = "sample_traffic_data.geojson"  # Replace with your geospatial file path
    map_file = "traffic_visualization.html"

    data = load_data(file_path)
    data = preprocess_data(data)
    features, labels = feature_engineering(data)
    model = train_model(features, labels)
    visualize_data(data, map_file)

if __name__ == "__main__":
    main()
