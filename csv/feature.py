import pandas as pd

def generate_geojson_features(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Initialize features list with the city center point
    features = [{
        "type": "Feature",
        "properties": {
            "name": "City Center",
            "popupContent": "Lahore City Center",
            "marker-symbol": "star",
            "marker-color": "#ff0000"
        },
        "geometry": {
            "type": "Point",
            "coordinates": [74.3087, 31.5204]
        }
    }]
    
    # Generate feature for each coordinate pair
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {
                "popupContent": "Property Location",
                "marker-color": "#3388ff"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['Lng']), float(row['Lat'])]
            }
        }
        features.append(feature)
    
    # Print the features in GeoJSON format
    print("    {")
    print('      "type": "FeatureCollection",')
    print('      "features": [')
    
    # Print each feature
    for i, feature in enumerate(features):
        print("        {")
        print('          "type": "Feature",')
        print('          "properties": {')
        props = list(feature["properties"].items())
        for j, (key, value) in enumerate(props):
            comma = "," if j < len(props) - 1 else ""
            print(f'            "{key}": "{value}"{comma}')
        print("          },")
        print('          "geometry": {')
        print('            "type": "Point",')
        print(f'            "coordinates": {feature["geometry"]["coordinates"]}')
        print("          }")
        comma = "," if i < len(features) - 1 else ""
        print(f"        {comma}")
    
    print("      ]")
    print("    }")

# Usage
if __name__ == "__main__":
    csv_path = "coordinates.csv"  # Replace with your CSV file path
    generate_geojson_features(csv_path)
