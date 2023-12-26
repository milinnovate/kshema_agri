import os
import calendar
from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads as wkt_loads
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, Geometry, CRS
import matplotlib.pyplot as plt

# Configure Sentinel Hub
config = SHConfig()

config.instance_id = '398f0429-2e8c-47cf-94a5-dc1efa6e83b4'
config.sh_client_id = '2b0c6570-a03d-4188-b1c6-928915874487'
config.sh_client_secret = 'hn6orfnDK2HeyHdf0PmJJfGT6WPO4jKu'

# Load CSV data containing polygons and information
csv_file_path = 'data.csv'
data = pd.read_csv(csv_file_path)

# Convert WKT to geometry objects and create GeoDataFrame
data['geometry'] = data['WKT'].apply(wkt_loads)
gdf = gpd.GeoDataFrame(data, geometry='geometry')

# NDVI calculation function
def calculate_ndvi(red, nir):
    return (nir - red) / (nir + red)

# Function to fetch Sentinel imagery
def fetch_sentinel_imagery(geometry, start_date, end_date, size):
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
        config=config,
    )
    response = request_rgb.get_data()
    return response[0] if response else None

# Function to process and save images for each polygon
def process_and_save_images(polygon_id, start_date, end_date, folder):
    polygon_geometry = gdf.loc[polygon_id]['geometry']
    polygon_sentinel = Geometry(polygon_geometry, CRS.WGS84)

    # NDVI Image
    evalscript_ndvi = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08"],
                output: { bands: 1 }
            };
        }
        function evaluatePixel(sample) {
            let ndvi = index(sample.B08, sample.B04);
            return [ndvi];
        }
    """
    request_ndvi = SentinelHubRequest(
        evalscript=evalscript_ndvi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(start_date, end_date),
            )
        ],
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
        geometry=polygon_sentinel,
        config=config,
    )
    ndvi_image = request_ndvi.get_data()[0]

    # Save NDVI image
    plt.imshow(ndvi_image, cmap='RdYlGn')
    plt.axis('off')
    ndvi_filename = os.path.join(folder, f'Polygon_{polygon_id}_NDVI_{start_date}_{end_date}.png')
    plt.savefig(ndvi_filename)
    plt.close()

    # True Color Image
    true_color_image = fetch_sentinel_imagery(polygon_sentinel, start_date, end_date, betsiboka_size)

    # Save True Color image
    if true_color_image is not None:
        plt.imshow(true_color_image)
        plt.axis('off')
        true_color_filename = os.path.join(folder, f'Polygon_{polygon_id}_TrueColor_{start_date}_{end_date}.png')
        plt.savefig(true_color_filename)
        plt.close()

# Ensure the output directory exists
output_folder = 'saved_images'
os.makedirs(output_folder, exist_ok=True)

# Iterate over polygons and a predefined time range
for polygon_id in gdf.index:
    for month in range(1, 13):  # Assuming you want to process all months
        year = 2023  # Specify the year
        last_day = calendar.monthrange(year, month)[1]
        start_date_str = datetime(year, month, 1).strftime('%Y-%m-%d')
        end_date_str = datetime(year, month, last_day).strftime('%Y-%m-%d')

        process_and_save_images(polygon_id, start_date_str, end_date_str, output_folder)
