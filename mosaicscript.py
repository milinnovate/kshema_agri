import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, Geometry, CRS
from shapely.wkt import loads as wkt_loads
import calendar

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

# Function to fetch NDVI image data for a given polygon and date range
def fetch_ndvi_image_data(polygon_id, start_date, end_date, resolution=10):
    polygon_geometry = gdf.loc[polygon_id]['geometry']
    polygon_sentinel = Geometry(polygon_geometry, CRS.WGS84)

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
        size=[resolution, resolution]  # Specify resolution here
    )
    response = request_ndvi.get_data()
    return response[0] if response else None

# Function to create and save NDVI mosaics for each month
def create_and_save_monthly_mosaics(year, dpi=300):
    output_folder = 'mosaic_images'
    os.makedirs(output_folder, exist_ok=True)

    for month in range(1, 13):  # For each month
        ndvi_images = []
        last_day = calendar.monthrange(year, month)[1]
        start_date = datetime(year, month, 1).strftime('%Y-%m-%d')
        end_date = datetime(year, month, last_day).strftime('%Y-%m-%d')

        for polygon_id in gdf.index:
            ndvi_image = fetch_ndvi_image_data(polygon_id, start_date, end_date)
            if ndvi_image is not None:
                ndvi_images.append(ndvi_image)

        # Create mosaic for the month if there are images
        if ndvi_images:
            mosaic = np.concatenate(ndvi_images, axis=1)  # Horizontal concatenation
            plt.imshow(mosaic, cmap='RdYlGn')
            plt.axis('off')
            plt.title(f'NDVI Mosaic - {year}-{month:02d}')

            # Save mosaic with high resolution
            output_path = os.path.join(output_folder, f'ndvi_mosaic_{year}-{month:02d}.png')
            plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
            plt.close()
            print(f"Mosaic for {year}-{month:02d} saved to {output_path}")

# Create and save monthly mosaics for a specified year
create_and_save_monthly_mosaics(2023)