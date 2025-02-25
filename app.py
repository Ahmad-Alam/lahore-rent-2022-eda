import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_folium import folium_static

# Page configuration
st.set_page_config(
    page_title="Lahore Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv("csv/lahore-property-rents-preprocessed.csv")
    return df

def create_model(df):
    # Prepare features
    features = ['Type', 'Area_Marla', 'Beds', 'Baths', 'Lat', 'Lng']
    X = df[features]
    y = df['Price']
    
    # Convert categorical variables
    X = pd.get_dummies(X, columns=['Type'])
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns

def main():
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Market Overview", "Property Analysis", "Price Predictor", "Location Comparison"])
    
    if page == "Market Overview":
        st.title("Lahore Property Market Overview 🏘️")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            most_common_type = df['Type'].mode()[0]
            type_percentage = (df['Type'] == most_common_type).mean() * 100
            st.metric("Most Common Property", 
                      f"{most_common_type}",
                      f"{type_percentage:.1f}% of market")

        with col2:
            price_range = f"PKR {df['Price'].quantile(0.25):,.0f} - {df['Price'].quantile(0.75):,.0f}"
            st.metric("Typical Price Range", 
                      price_range,
                      "25th-75th percentile")

        with col3:
            total_properties = len(df)
            monthly_properties = total_properties // 12  # Assuming 1 year of data
            st.metric("Market Activity",
                      f"{total_properties:,} listings",
                      f"~{monthly_properties:,} per month")

        with col4:
            popular_areas = df['Location'].value_counts().head(3).index.tolist()
            st.metric("Top Areas",
                      f"{popular_areas[0]}",
                      f"+ {popular_areas[1:3]}")        
        # Property type distribution
        st.subheader("Property Type Distribution")
        fig = px.pie(df, names='Type', title='Distribution of Property Types',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)
        
        # Price distribution
        st.subheader("Price Distribution by Property Type")
        fig = px.box(df, x='Type', y='Price', title='Price Distribution by Property Type')
        st.plotly_chart(fig)
        
        # Map visualization
        st.subheader("Geographical Price Distribution")
        map_center = [31.5204, 74.3587]  # Lahore coordinates
        m = folium.Map(location=map_center, zoom_start=11)
        
        # Sample data for better visualization
        sample_df = df.sample(n=min(1000, len(df)))
        
        for idx, row in sample_df.iterrows():
            folium.CircleMarker(
                location=[row['Lat'], row['Lng']],
                radius=5,
                popup=f"Price: PKR {row['Price']:,}<br>Type: {row['Type']}",
                color='red',
                fill=True
            ).add_to(m)
        
        folium_static(m)
        
    elif page == "Property Analysis":
        st.title("Property Analysis Dashboard 📊")
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = ['Price', 'Area_Marla', 'Beds', 'Baths']
        corr = df[numeric_cols].corr()
        
        # Create heatmap using go.Figure
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=numeric_cols,
            y=numeric_cols,
            text=corr.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdBu'
        ))
        
        fig.update_layout(
            title='Correlation Heatmap',
            width=600,
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Area vs Price scatter plot
        st.subheader("Area vs Price Relationship")
        fig = px.scatter(df, x='Area_Marla', y='Price', color='Type',
                        title='Property Price vs Area')
        st.plotly_chart(fig)
        
        # Room configuration analysis
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Bedroom Distribution")
            fig = px.histogram(df, x='Beds', title='Distribution of Bedrooms')
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Bathroom Distribution")
            fig = px.histogram(df, x='Baths', title='Distribution of Bathrooms')
            st.plotly_chart(fig)        
    elif page == "Price Predictor":  # Price Predictor page
        st.title("Property Price Predictor 💰")
        
        # Create two columns for input
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            property_type = st.selectbox("Property Type", df['Type'].unique())
            area = st.number_input("Area (Marlas)", min_value=1, max_value=50, value=10)
            beds = st.number_input("Number of Bedrooms", min_value=1, max_value=7, value=3)
        
        with input_col2:
            baths = st.number_input("Number of Bathrooms", min_value=1, max_value=7, value=2)
            
            # Get unique locations for autocomplete
            unique_locations = sorted(df['Location'].unique())
            
            # Create searchable dropdown for locations
            location = st.selectbox(
                "Select Location",
                unique_locations,
                key="location_select"
            )
            st.write("")
            st.write("")
            
            # Predict Price button directly in input_col2
            predict_clicked = st.button("Predict Price", use_container_width=True)

        # Create two columns for map and results
        map_col, results_col = st.columns([1, 1])
        
        with map_col:
            # Get the coordinates for selected location
            location_coords = df[df['Location'] == location][['Lat', 'Lng']].iloc[0]
            lat, lng = location_coords['Lat'], location_coords['Lng']
            
            # Display selected location on map
            st.subheader("Selected Location")
            m = folium.Map(location=[lat, lng], zoom_start=13)
            folium.Marker(
                [lat, lng],
                popup=location,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            folium_static(m)
        
        with results_col:
            if predict_clicked:
                # Load or create model
                model, columns = create_model(df)
                
                # Prepare input data
                input_data = pd.DataFrame({
                    'Area_Marla': [area],
                    'Beds': [beds],
                    'Baths': [baths],
                    'Lat': [lat],
                    'Lng': [lng],
                    'Type': [property_type]
                })
                
                # Convert to model format
                input_data = pd.get_dummies(input_data, columns=['Type'])
                
                # Align input features with model features
                for col in columns:
                    if col not in input_data.columns:
                        input_data[col] = 0
                
                input_data = input_data[columns]
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.markdown("### Estimated Price")
                st.markdown(f"<div style='background-color: #1a472a; padding: 10px; border-radius: 5px;'>"
                           f"PKR {prediction:,.2f}"
                           f"</div>", unsafe_allow_html=True)
                
                # Show similar properties
                st.markdown("### Similar Properties")
                similar = df[
                    (df['Type'] == property_type) &
                    (df['Area_Marla'].between(area-2, area+2)) &
                    (df['Beds'] == beds)
                ].head()
                
                if not similar.empty:
                    # Style the dataframe
                    st.dataframe(
                        similar[['Type', 'Price', 'Area_Marla', 'Beds', 'Baths', 'Location']],
                        height=200,  # Adjust height to fit better
                        use_container_width=True
                    )
                else:
                    st.info("No similar properties found in the database.")
    elif page == "Location Comparison":
        st.title("Location Comparison Tool 📊")
        
        # Allow selection of 2-3 locations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location1 = st.selectbox(
                "Select First Location",
                sorted(df['Location'].unique()),
                key="location1"
            )
            
        with col2:
            # Filter out first location from second dropdown
            remaining_locations = sorted(loc for loc in df['Location'].unique() if loc != location1)
            location2 = st.selectbox(
                "Select Second Location",
                remaining_locations,
                key="location2"
            )
            
        with col3:
            # Optional third location
            remaining_locations2 = sorted(loc for loc in remaining_locations if loc != location2)
            location3 = st.selectbox(
                "Select Third Location (Optional)",
                ["None"] + remaining_locations2,
                key="location3"
            )
        
        # Get data for each location
        loc1_data = df[df['Location'] == location1]
        loc2_data = df[df['Location'] == location2]
        loc3_data = df[df['Location'] == location3] if location3 != "None" else None
        
        # Function to get location statistics
        def get_location_stats(data):
            stats = {
                'avg_price': data['Price'].mean(),
                'min_price': data['Price'].min(),
                'max_price': data['Price'].max(),
                'property_types': data['Type'].value_counts(),
                'avg_area': data['Area_Marla'].mean(),
                'common_sizes': data['Area_Marla'].value_counts().head(3),
                'bed_config': data['Beds'].value_counts().head(3),
                'bath_config': data['Baths'].value_counts().head(3),
                'total_properties': len(data)
            }
            return stats
        
        # Calculate statistics for each location
        stats1 = get_location_stats(loc1_data)
        stats2 = get_location_stats(loc2_data)
        stats3 = get_location_stats(loc3_data) if loc3_data is not None else None
        
        # Display comparisons
        st.subheader("Price Comparison")
        price_comparison = pd.DataFrame({
            location1: [stats1['avg_price'], stats1['min_price'], stats1['max_price']],
            location2: [stats2['avg_price'], stats2['min_price'], stats2['max_price']]
        }, index=['Average Price', 'Minimum Price', 'Maximum Price'])
        
        if stats3:
            price_comparison[location3] = [stats3['avg_price'], stats3['min_price'], stats3['max_price']]
        
        # Format prices
        price_comparison = price_comparison.applymap(lambda x: f"PKR {x:,.0f}")
        st.dataframe(price_comparison, use_container_width=True)
        
        # Property Type Distribution
        st.subheader("Property Type Distribution")
        fig = go.Figure()
        
        locations_data = [(location1, stats1), (location2, stats2)]
        if stats3:
            locations_data.append((location3, stats3))
        
        for loc_name, stats in locations_data:
            fig.add_trace(go.Bar(
                name=loc_name,
                x=stats['property_types'].index,
                y=stats['property_types'].values,
                text=stats['property_types'].values,
                textposition='auto',
            ))
        
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Common Property Sizes
        st.subheader("Common Property Sizes (Marla)")
        size_comparison = pd.DataFrame({
            location1: [f"{size:.1f} Marla ({count} properties)" 
                       for size, count in stats1['common_sizes'].items()],
            location2: [f"{size:.1f} Marla ({count} properties)" 
                       for size, count in stats2['common_sizes'].items()]
        }, index=['Most Common', '2nd Most Common', '3rd Most Common'])
        
        if stats3:
            size_comparison[location3] = [f"{size:.1f} Marla ({count} properties)" 
                                        for size, count in stats3['common_sizes'].items()]
        
        st.dataframe(size_comparison, use_container_width=True)
        
        # Room Configurations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bedroom Configuration")
            bed_comparison = pd.DataFrame({
                location1: [f"{bed} beds ({count} properties)" 
                           for bed, count in stats1['bed_config'].items()],
                location2: [f"{bed} beds ({count} properties)" 
                           for bed, count in stats2['bed_config'].items()]
            }, index=['Most Common', '2nd Most Common', '3rd Most Common'])
            
            if stats3:
                bed_comparison[location3] = [f"{bed} beds ({count} properties)" 
                                           for bed, count in stats3['bed_config'].items()]
            
            st.dataframe(bed_comparison, use_container_width=True)
        
        with col2:
            st.subheader("Bathroom Configuration")
            bath_comparison = pd.DataFrame({
                location1: [f"{bath} baths ({count} properties)" 
                           for bath, count in stats1['bath_config'].items()],
                location2: [f"{bath} baths ({count} properties)" 
                           for bath, count in stats2['bath_config'].items()]
            }, index=['Most Common', '2nd Most Common', '3rd Most Common'])
            
            if stats3:
                bath_comparison[location3] = [f"{bath} baths ({count} properties)" 
                                            for bath, count in stats3['bath_config'].items()]
            
            st.dataframe(bath_comparison, use_container_width=True)
        
        # Price Distribution
        st.subheader("Price Distribution Comparison")
        fig = go.Figure()
        
        for loc_name, data in [(location1, loc1_data), (location2, loc2_data)]:
            fig.add_trace(go.Box(
                y=data['Price'],
                name=loc_name,
                boxpoints='outliers'
            ))
        
        if loc3_data is not None:
            fig.add_trace(go.Box(
                y=loc3_data['Price'],
                name=location3,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            yaxis_title="Price (PKR)",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
