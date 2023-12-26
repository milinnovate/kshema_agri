import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Polygon
import os
from PIL import Image
import leafmap.foliumap as leafmap

import datetime 
from sentinelhub import SHConfig, SentinelHubRequest, MimeType, DataCollection, Geometry
from sentinelhub import Geometry, CRS
import matplotlib.pyplot as plt
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)


# Load your CSV data containing polygons and information
csv_file_path = 'all_together.csv'
data = pd.read_csv(csv_file_path)
data['geometry'] = data['WKT'].apply(wkt_loads)
gdf = gpd.GeoDataFrame(data)
# Filter out rows with empty geometries
gdf = gdf[~gdf['geometry'].is_empty]

if 'selected_polygon' not in st.session_state:
    st.session_state['selected_polygon'] = None


# Create a Streamlit web app
st.title('Agri-Watch Report : Executive Summary')


polygon_labels = [i + 1 for i in gdf.index]
selected_polygon_label = st.sidebar.selectbox('Select a Polygon', polygon_labels)

resolution = 60
betsiboka_coords_wgs84 = (46.16, -16.15, 46.51, -15.58)


betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

# Create a GeoDataFrame from the CSV data with WKT geometries

config = SHConfig()

config.instance_id = '7c65dff9-a0e0-40a9-889c-31947b4465e5'
config.sh_client_id = '0f8fe328-6a58-4210-bad0-4dc7886dfeef'
config.sh_client_secret = '5PxoFxGgWimanpy1SOtiXwLNX0sgezOC'


output_csv_path = 'output.csv'
output_data = pd.read_csv(output_csv_path)

def style_data(df):
    return df.style.set_properties(**{
        'background-color': 'black',
        'color': 'lime',
        'font-size' : '50pt',
        'font-weight': 'bold',
        'border-color': 'white'
    })

def calculate_ndvi(red, nir):
    return (nir - red) / (nir + red)

def fetch_sentinel_imagery(geometry, start_date, end_date, size):
    # You can modify this evalscript to fetch the desired imagery
    evalscript_rgb = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04"],
                output: { bands: 3 }
            };
        }
        function evaluatePixel(sample) {
            return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
        }
    """

    request_rgb = SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(start_date, end_date),
            )
        ],
        responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
        geometry=geometry,
        size=size,
        config=config,
    )

    response = request_rgb.get_data()
    return response[0] if response else None

evalscript_all_bands = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
            }],
            output: {
                bands: 13
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12];
    }
"""
def get_polygon_bounds(geometry):
    minx, miny, maxx, maxy = geometry.bounds
    return [[miny, minx], [maxy, maxx]]



def fetch_dem_imagery(geometry, start_date, end_date, size):
    evalscript_dem = """
        // DEM Topographic Visualization
        //VERSION=3
        function setup() {
            return {
                input: ["DEM"],
                output: { bands: 3 }
            };
        }
function evaluatePixel(sample) {

  let val = sample.DEM;
  let imgVals = null;

  if (val > 8000) imgVals = [1, 1, 1];
  else if (val > 7000) imgVals = [0.95, 0.95, 0.95];
  else if (val > 6000) imgVals = [0.9, 0.9, 0.9];
  else if (val > 5500) imgVals = [0.29, 0.22, 0.07];
  else if (val > 5000) imgVals = [0.37, 0.30, 0.15];
  else if (val > 4500) imgVals = [0.45, 0.38, 0.22];
  else if (val > 4000) imgVals = [0.53, 0.45, 0.30];
  else if (val > 3500) imgVals = [0.6, 0.53, 0.38];
  else if (val > 3000) imgVals = [0.68, 0.61, 0.46];
  else if (val > 2500) imgVals = [0.76, 0.69, 0.54];
  else if (val > 2000) imgVals = [0.84, 0.77, 0.62];
  else if (val > 1500) imgVals = [0.92, 0.85, 0.69];
  else if (val > 1000) imgVals = [0.99, 0.93, 0.75];
  else if (val > 900) imgVals = [0.67, 0.87, 0.63];
  else if (val > 800) imgVals = [0.65, 0.84, 0.61];
  else if (val > 700) imgVals = [0.59, 0.81, 0.56];
  else if (val > 600) imgVals = [0.52, 0.76, 0.48];
  else if (val > 500) imgVals = [0.48, 0.73, 0.44];
  else if (val > 400) imgVals = [0.45, 0.70, 0.40];
  else if (val > 300) imgVals = [0.37, 0.64, 0.33];
  else if (val > 200) imgVals = [0.30, 0.58, 0.25];
  else if (val > 100) imgVals = [0.24, 0.53, 0.24];
  else if (val > 75) imgVals = [0.21, 0.49, 0.23];
  else if (val > 50) imgVals = [0.18, 0.45, 0.18];
  else if (val > 25) imgVals = [0.15, 0.41, 0.13];
  else if (val > 10) imgVals = [0.12, 0.37, 0.08];
  else if (val > 0) imgVals = [0.09, 0.33, 0.03];
  else imgVals = [0.06, 0.06, 0.55];

  return imgVals;
        }
    """

    request_dem = SentinelHubRequest(
        evalscript=evalscript_dem,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.DEM,
                time_interval=(start_date, end_date),
            )
        ],
        responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
        geometry=geometry,
        config=config,
    )

    response = request_dem.get_data()
    return response[0] if response else None


# Function to validate and potentially correct polygon geometry
def validate_and_correct_geometry(geometry):
    try:
        return geometry.buffer(0)
    except Exception as e:
        st.warning(f"Geometry validation error: {e}")
        return Polygon()


# Adjust the index to access the correct polygon from gdf
selected_polygon = selected_polygon_label - 1

start_date = st.sidebar.date_input("Select start date", datetime.date(2022, 6, 1))
end_date = st.sidebar.date_input("Select end date", datetime.date(2022, 7, 1))
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")
if end_date < start_date:
    st.sidebar.error("End date cannot be before start date.")
    valid_dates = False
else:
    valid_dates = True


# Filter the GeoDataFrame to get the selected polygon
selected_polygon_data = gdf.loc[selected_polygon]

# Validate and potentially correct the selected polygon's geometry
valid_geometry = validate_and_correct_geometry(selected_polygon_data['geometry'])



# Filter the GeoDataFrame to get the selected polygon

# Validate and potentially correct the selected polygon's geometry



new_polygon = valid_geometry
polygon_wkt = new_polygon.wkt
new_polygon_sentinel = Geometry(wkt_loads(polygon_wkt), CRS.WGS84)

resolution = 60
betsiboka_coords_wgs84 = (46.16, -16.15, 46.51, -15.58)

betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)




# Update the request to use the selected dates
request_all_bands = SentinelHubRequest(
    evalscript=evalscript_all_bands,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=(start_date_str, end_date_str),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    geometry=new_polygon_sentinel,
    size=betsiboka_size,
    config=config,
)



evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            maxcc=0.3,
            mosaicking_order=MosaickingOrder.LEAST_CC,
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2023-08-01", "2023-12-15"),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    geometry=new_polygon_sentinel,
    config=config,
)

def style_dataframe(df):
    return df.style.set_properties(
        **{
            'font-size': '25pt',  # Increase font size (adjust as needed)
            'font-weight': 'bold' # Make font bold
        }
    ).applymap(
        lambda x: 'color: green;' if x is True else ('color: red;' if x is False else '')
    )


def split_dataframe(df):
    half_point = len(df) // 2  # Find the midpoint to split the DataFrame
    df1 = df.iloc[:half_point]  # First half
    df2 = df.iloc[half_point:]  # Second half
    return df1, df2

# Check if the geometry is valid
if not valid_geometry.is_valid:
    st.warning("Selected polygon has an invalid geometry and cannot be displayed.")
else:

    high_res_folder = 'high_res'  # Replace with the path to your high_res folder
    image_path = os.path.join(high_res_folder, f'{selected_polygon + 1}.png')  # Assuming polygon index starts from 0

    if os.path.exists(image_path):
        st.image(Image.open(image_path), caption=f'High-Resolution Image for Polygon {selected_polygon + 1}', use_column_width=True)
    else:
        st.error(f"High-resolution image not found for Polygon {selected_polygon}")

    bounds = get_polygon_bounds(valid_geometry)
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = leafmap.Map(center=center, zoom_start=12, basemap='SATELLITE')
    # folium.TileLayer('https://{s}.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    #                  attr='Esri',
    #                  name='Esri WorldImagery',
    #                  control=False).add_to(m)

    m.fit_bounds(bounds)
    selected_polygon_output_data = output_data[output_data['poly'] == selected_polygon + 1]

# Display the data for the selected polygon

# Display the data for the selected polygon in a four-column layout
    if not selected_polygon_output_data.empty:
        st.write(f"Data for Polygon {selected_polygon + 1}:")


        # Splitting the DataFrame into two sub-DataFrames
        df1, df2 = split_dataframe(selected_polygon_output_data.transpose())

        # Creating two columns and displaying each styled sub-DataFrame
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(style_dataframe(df1), use_container_width=True)
        with col2:
            st.dataframe(style_dataframe(df2), use_container_width=True)

    else:
        st.error(f"No data found for Polygon {selected_polygon + 1} in 'output.csv'")

    # Convert geometry to GeoJSON for Folium with transparent fill
    geo_json = folium.GeoJson(
        data=valid_geometry.__geo_interface__,
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 2,
            'fillOpacity': 0  # Adjust this value for desired transparency
        }
    )
    geo_json.add_to(m)

    # Add marker
    folium.Marker([valid_geometry.centroid.y, valid_geometry.centroid.x], tooltip=f'Polygon {selected_polygon}').add_to(m)

    # Sentinel Imagery


    folium_static(m)

if valid_dates:

    if st.sidebar.button('Submit'):

        # Retrieve data for all bands
        all_bands_imgs = request_all_bands.get_data()


        # Calculate NDVI
        red_band = all_bands_imgs[0][:, :, 3]
        nir_band = all_bands_imgs[0][:, :, 7]
        ndvi = calculate_ndvi(red_band, nir_band)

        # Display NDVI plot
        plt.figure()
        plt.imshow(ndvi, cmap='RdYlGn', aspect="auto")
        plt.title(f"NDVI ({start_date.strftime('%B')} to {end_date.strftime('%B')} {start_date.year})")
        plt.axis('off')
        plt.show()

        # Display NDVI plot in Streamlit
        st.pyplot(plt)

        true_color_imgs = request_true_color.get_data()

        # Display the true color image
        if true_color_imgs:
            image = true_color_imgs[0]
            plt.figure()
            plt.imshow(image * 3.5 / 255, aspect="auto")
            plt.axis('off')
            plt.title("True Color Imagery")
            st.pyplot(plt)
        else:
            st.warning("No true color imagery available for the selected polygon and time range.")

        dem_image = fetch_dem_imagery(new_polygon_sentinel, start_date_str, end_date_str, betsiboka_size)
        if dem_image is not None:
            st.write("DEM Topographic Visualization:")
            plt.figure()
            plt.imshow(dem_image)
            plt.axis('off')
            plt.show()
            st.pyplot(plt)
        else:
            st.warning("No DEM imagery available for the selected polygon and time range.")


logo_path = 'logo.png'  # Replace with the path to your logo file
st.sidebar.image(logo_path, use_column_width=True)
