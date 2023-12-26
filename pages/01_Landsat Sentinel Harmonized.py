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

# Load output.csv data
output_data = pd.read_csv('output.csv')
# Adjusting the index to start from 1 for user display
polygon_options = [i + 1 for i in gdf.index]

# Initialize 'selected_polygon' in session state if it's not already present
if 'selected_polygon' not in st.session_state:
    st.session_state['selected_polygon'] = polygon_options[0]

# Function to load and display images for a given polygon
def load_and_display_images_and_text(polygon_id, folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.startswith(f"{polygon_id}_"):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)

            # Creating two columns for image and text
            col1, col2 = st.columns([3, 2])  # Adjust the ratio as needed

            with col1:
                st.image(image, caption=file_name, use_column_width=True)

            with col2:
                # Extracting and displaying relevant data from output.csv
                selected_data = output_data[output_data['poly'] == polygon_id]
                if 'NDMI' in file_name:
                    # Display ndmi_status and sar_water_index
                    st.text("NDMI Status: " + str(selected_data['ndmi_status'].values[0]))
                    st.text("SAR Water Index: " + str(selected_data['sar_water_index'].values[0]))
                elif 'NDVI' in file_name:
                    # Display cropstage_on_event and other relevant columns
                    st.text("Crop Stage on Event: " + str(selected_data['cropstage_on_event'].values[0]))
                    st.text("Phenology Similar: " + str(selected_data['phenology_similar'].values[0]))
                    st.text("Phenology Same as 5 Year: " + str(selected_data['phenology_same_as_5year'].values[0]))
                    st.text("Harvest Date: " + str(selected_data['harvest_date'].values[0]))
                elif 'NDWI' in file_name:
                    # Display cropstage_on_event and other relevant columns
                    st.text("Precipitation Analysis: " + str(selected_data['ppt_ana'].values[0]))


# Sidebar for polygon selection
selected_polygon_label = st.sidebar.selectbox('Select a Polygon', polygon_options)

# Adjusting the index to match internal data (subtracting 1)
selected_polygon = selected_polygon_label - 1
st.session_state['selected_polygon'] = selected_polygon

# Define the path to the HLS folder
hls_folder_path = 'media_package/HLS'  # Replace with your HLS folder path

st.title('HLS Plots')

# Display images for the selected polygon
st.header(f'Displaying Plots for Polygon {selected_polygon_label}')
load_and_display_images_and_text(selected_polygon_label, hls_folder_path)
logo_path = 'logo.png'  # Replace with the path to your logo file
st.sidebar.image(logo_path, use_column_width=True)
logo_path = 'logo.png'  # Replace with the path to your logo file
st.sidebar.image(logo_path, use_column_width=True)
