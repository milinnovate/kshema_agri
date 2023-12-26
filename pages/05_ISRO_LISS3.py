import streamlit as st
import os
from PIL import Image
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads as wkt_loads

# Load your CSV data containing polygons and information
csv_file_path = 'all_together.csv'
data = pd.read_csv(csv_file_path)
data['geometry'] = data['WKT'].apply(wkt_loads)
gdf = gpd.GeoDataFrame(data)

# Adjusting the index to start from 1 for user display
polygon_options = [i + 1 for i in gdf.index]

# Initialize 'selected_polygon' in session state if it's not already present
if 'selected_polygon' not in st.session_state:
    st.session_state['selected_polygon'] = polygon_options[0]

# Function to load and display images for a given polygon
def load_and_display_images(polygon_id, folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.startswith(f"{polygon_id}_"):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            st.image(image, caption=file_name, use_column_width=True)

# Sidebar for polygon selection
selected_polygon_label = st.sidebar.selectbox('Select a Polygon', polygon_options)

# Adjusting the index to match internal data (subtracting 1)
selected_polygon = selected_polygon_label - 1
st.session_state['selected_polygon'] = selected_polygon

# Define the path to the HLS folder
hls_folder_path = 'media_package/ISRO'  # Replace with your HLS folder path

st.title('ISRO Plots')

# Display images for the selected polygon
st.header(f'Displaying Plots for Polygon {selected_polygon_label}')
load_and_display_images(selected_polygon_label, hls_folder_path)
logo_path = 'logo.png'  # Replace with the path to your logo file
st.sidebar.image(logo_path, use_column_width=True)
