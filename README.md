# Lahore Property Market Analysis & Price Predictor

Analysis and interactive exploration of the rental property market in Lahore, Pakistan using Python, data science techniques, and an interactive Streamlit web application. The project explores pricing patterns, market segmentation, and geographical distributions through statistical analysis, visualization, and machine learning prediction.

## Features
- Property price analysis and market segmentation
- Geocoding implementation using Google Maps API
- Advanced visualizations with seaborn, matplotlib, and Plotly
- Statistical analysis of property features
- Spatial distribution analysis with Folium maps
- Price-per-area calculations and correlations
- Interactive Streamlit web application with four specialized modules
- Machine learning-based price prediction using Random Forest

## Tools Used
- Python
- pandas
- seaborn & matplotlib
- Plotly & Folium
- Google Maps API
- numpy
- Streamlit
- scikit-learn

## Application Components
- **Market Overview**: Key metrics dashboard, property distribution visualizations, and geographic mapping
- **Property Analysis**: Correlation analysis, price vs area relationships, and room configuration statistics
- **Price Predictor**: ML-based price estimation with Random Forest and similar property comparison
- **Location Comparison**: Multi-location analysis for side-by-side comparison of different areas

## Analysis Highlights
- Property type distribution and pricing patterns
- Room configuration analysis
- Geographical price distribution
- Feature correlation studies
- Outlier detection and handling
- Predictive modeling of property prices

## Data Source
Dataset used in this analysis is from Kaggle: [Lahore Property Rents](https://www.kaggle.com/datasets/sherafgunmetla/lahore-property-rents) by Sher Afgun Metla.

## Getting Started
1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Add your Google Maps API key in the notebook (for analysis) or the app.py file (for the app)
4. Run the Jupyter notebook for analysis or launch the Streamlit app with `streamlit run app.py`

Note: The Google Maps API key in the notebook has been removed for security. You'll need to add your own key to run the geocoding cells. But no need to worry, the processed csv is already present in the csv folder for use.

## Running the Streamlit App
```
streamlit run app.py
```
This will launch the interactive web application with all four components accessible via the sidebar navigation.
