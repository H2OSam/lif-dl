import numpy as np
import cartopy
from cartopy.io.img_tiles import GoogleTiles
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.crs import PlateCarree
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patheffects as patheffects
from geopy import distance
from geopy.distance import distance as geopy_distance

# Global font settings for the entire figure
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

## HELPER FUNCTIONS
# Custom scalebar function
def displace(lat, lon, bearing, distance_km):
    """
    Calculate new lat/lon given starting point, bearing (degrees), and distance (meters)
    """    
    # Use geopy to calculate destination point
    origin = (lat, lon)
    destination = geopy_distance(kilometers=distance_km).destination(origin, bearing)
    
    return destination.latitude, destination.longitude

def add_custom_scalebar(ax, projection):
    """
    Add a custom scalebar to the map using Lambert Conformal projection coordinates
    """
    # Get the extent of the current axes in projection coordinates (meters)
    x0, x1, y0, y1 = ax.get_extent(projection)
    
    # Scale bar parameters - positioned in bottom right corner
    bar_offset = [0.95, 0.05, 0.07]  # x, y position and text height offset
    
    # Scale bar settings
    bar_tickmark = 100000  # 100 km per segment (in meters)
    bar_ticks = 5
    total_distance_m = bar_ticks * bar_tickmark
    
    # Calculate scalebar width in projection coordinates (meters)
    scalebar_width_m = total_distance_m
    
    # Position scalebar so it ends at 95% from left edge
    bar_x1 = x0 + (x1-x0)*bar_offset[0]  # Right edge of scalebar
    bar_x0 = bar_x1 - scalebar_width_m   # Left edge of scalebar
    bar_y0 = y0 + (y1-y0)*bar_offset[1]  # Bottom position
    
    text_x0 = bar_x0
    text_y0 = y0 + (y1-y0)*bar_offset[2]
    
    bar_color = ['black', 'white']
    
    # Draw the scale bar segments
    current_x = bar_x0
    current_y = bar_y0
    
    for i in range(bar_ticks):
        # Calculate end position (horizontal bar moving eastward)
        end_x = current_x + bar_tickmark
        end_y = current_y

        # Draw line segment with alternating colors
        ax.plot([current_x, end_x], [current_y, end_y],
                color=bar_color[i%2],
                linewidth=10, transform=projection,
                solid_capstyle='butt', zorder=10)
        
        # Start of next bar is end of last bar
        current_x = end_x
        current_y = end_y

    # Add scale bar text with white background
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    total_distance_km = total_distance_m / 1000
    units = 'km'

    ax.text(text_x0, text_y0, f'{int(total_distance_km)} {units}', 
            transform=projection,
            horizontalalignment='left', verticalalignment='bottom',
            path_effects=buffer, zorder=11, fontsize=12, fontweight='bold')

# Configuration
shapefile_dir = "/mnt/c/Users/johns/OneDrive - H2O Geomatics Inc/Documents/Reports Papers & Presentations/lake-ice-deep-learning/shapefiles/"

# Load shapefiles
# shp_110m_lake = cartopy.io.shapereader.Reader(shapefile_dir + "ne_110m_lakes/ne_110m_lakes.shp")
shp_10m_lake = cartopy.io.shapereader.Reader(shapefile_dir + "ne_10m_lakes/ne_10m_lakes.shp")
shp_study_lakes = cartopy.io.shapereader.Reader(shapefile_dir + "lakescci_v2.0.2/Lakes_for_Sam.shp")

# Pre-load and cache geometries
lake_geometries = list(shp_10m_lake.geometries())
study_lake_records = list(shp_study_lakes.records())
study_lake_geometries = [lake.geometry for lake in study_lake_records]

# Map shapefile names to canonical display names used in figure labels
shapefile_to_display_name = {
    "Great Bear": "Great Bear Lake",
    "Great Slave": "Great Slave Lake",
    "Athabasca": "Lake Athabasca",
    "Reindeer": "Reindeer Lake",
    "Winnipeg": "Lake Winnipeg",
}

def get_display_name(raw_name):
    return shapefile_to_display_name.get(raw_name, raw_name)

# Map lake names to numbers for labeling
lake_name_to_label = {
    "Great Bear Lake": "1.",
    "Great Slave Lake": "2.",
    "Lake Athabasca": "3.",
    "Reindeer Lake": "4.",
    "Lake Winnipeg": "5.",
}

# Color scheme
lake_color = "#8cb9d1"
land_color = "#fbf0e2"
highlight_color = "#ffdd44"
highlight_border = "#e1c33e"

# Create the figure with lower DPI for faster rendering
fig = plt.figure(figsize=(20, 10), dpi=300, facecolor='white')

# Create a top-level 1 row, 2 column gridspec layout
gs_main = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1, width_ratios=[1.5, 1])
projection = ccrs.LambertConformal(central_longitude=-95, central_latitude=60)

# Column 1: Overview map (takes the entire first column)
ax_canada = fig.add_subplot(gs_main[0, 0], projection=projection)

# Column 2: Create a nested gridspec for the lake sites (3 rows, 2 columns)
gs_lakes = gridspec.GridSpecFromSubplotSpec(3, 2, gs_main[0, 1],
                                            hspace=0.15, wspace=0.05)

# Define the lake site axes within the nested gridspec
ax_site1 = fig.add_subplot(gs_lakes[0, 0], projection=projection)
ax_site2 = fig.add_subplot(gs_lakes[0, 1], projection=projection)
ax_site3 = fig.add_subplot(gs_lakes[1, 0], projection=projection)
ax_site4 = fig.add_subplot(gs_lakes[1, 1], projection=projection)
ax_site5 = fig.add_subplot(gs_lakes[2, 0], projection=projection)
ax_site5_placeholder = fig.add_subplot(gs_lakes[2, 1], projection=projection)

# Center the 5th panel below the 2x2 grid and match its size to the top panels
top_panel_pos = ax_site1.get_position()
bottom_left_pos = ax_site5.get_position()
bottom_right_pos = ax_site5_placeholder.get_position()
centered_x0 = (bottom_left_pos.x0 + bottom_right_pos.x1 - top_panel_pos.width) / 2
ax_site5.set_position([centered_x0, bottom_left_pos.y0, top_panel_pos.width, top_panel_pos.height])
fig.delaxes(ax_site5_placeholder)

# Store axes in order for easy iteration
site_axes = [ax_site1, ax_site2, ax_site3, ax_site4, ax_site5]

# Setup Canada overview map
canada_extent = [-141, -52, 41, 84]  # [min_lon, max_lon, min_lat, max_lat]
ax_canada.set_extent(canada_extent, crs=ccrs.PlateCarree())

# Add map features to overview - use cached geometries and rasterization
ax_canada.add_feature(cartopy.feature.LAND, facecolor=land_color, rasterized=True)
ax_canada.add_geometries(lake_geometries, 
                         crs=ccrs.PlateCarree(), 
                         facecolor=lake_color, edgecolor='none', linewidth=0, rasterized=True)

# Add country boundaries - use 50m for faster rendering
ax_canada.add_feature(cartopy.feature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m'),
                      edgecolor='black', facecolor='none', linewidth=0.5, rasterized=True)

# Add study lakes with highlight
ax_canada.add_geometries(study_lake_geometries,
                         crs=ccrs.PlateCarree(),
                         facecolor=highlight_color, edgecolor=highlight_border, linewidth=1)

# Add the custom scalebar instead
add_custom_scalebar(ax_canada, projection)

# Add gridlines
gls = ax_canada.gridlines(draw_labels=True, 
                          xlocs=np.arange(-200, 10, 20),
                          ylocs=np.arange(30, 95, 5),
                          linewidth=0.5,
                          x_inline=False,
                          y_inline=False,
                          color='gray')
gls.xlabels_top = False
# Set the font size for grid labels
gls.xlabel_style = {'size': 10}

# Label offsets for lake numbers
lake_label_offset = {
    "Great Bear Lake": (-5.5, 0.5),
    "Great Slave Lake": (-5.5, 0.5),
    "Lake Athabasca": (-3.5, 0.5),
    "Reindeer Lake": (-3, 0.5),
    "Lake Winnipeg": (-3, 0.5),
}

# Add numbered labels to study lakes
for lake in study_lake_records:
    raw_name = lake.attributes['name']
    display_name = get_display_name(raw_name)
    name = lake_name_to_label[display_name]
    geometry = lake.geometry
    centroid = geometry.centroid
    x_offset, y_offset = lake_label_offset[display_name]
    ax_canada.text(centroid.x + x_offset, centroid.y + y_offset,
                   name, fontsize=20, ha='center', va='center',
                   transform=ccrs.PlateCarree(), color='black')

# Define extents for individual lake maps (in projected coordinates)
lake_extents = {
    "Great Bear Lake": (275000, 275000, 275000, 275000),
    "Great Slave Lake": (260000, 320000, 240000, 340000),
    "Lake Athabasca": (180000, 240000, 210000, 210000),
    "Lake Winnipeg": (275000, 275000, 310000, 240000),
    "Reindeer Lake": (140000, 140000, 140000, 140000),
}

# Sort the shapefile records in the desired order for the lake maps
desired_lake_order = ["Great Bear Lake", "Great Slave Lake", "Lake Athabasca", "Reindeer Lake", "Lake Winnipeg"]
records_by_name = {get_display_name(record.attributes['name']): record for record in study_lake_records}
ordered_lakes = [records_by_name[name] for name in desired_lake_order]

# Create individual lake maps
tiler= GoogleTiles(style="satellite")  # or another tile source
for ax, lake in zip(site_axes, ordered_lakes):
    raw_lake_name = lake.attributes['name']
    lake_name = get_display_name(raw_lake_name)
    
    # Get lake centroid and transform to projected coordinates
    centroid = lake.geometry.centroid
    lon_center, lat_center = centroid.x, centroid.y
    
    # Transform centroid from lat/lon to Lambert Conformal coordinates
    transformed = projection.transform_point(lon_center, lat_center, PlateCarree())
    x_center, y_center = transformed[0], transformed[1]
    
    # Get extent parameters for this lake
    x_extent_left, x_extent_right, y_extent_left, y_extent_right = lake_extents[lake_name]
    
    # Define extent in projected coordinates
    lake_extent = [
        x_center - x_extent_left,   # left
        x_center + x_extent_right,  # right  
        y_center - y_extent_left,   # bottom
        y_center + y_extent_right   # top
    ]
    
    # Set the extent for this lake map
    ax.set_extent(lake_extent, crs=projection)
    ax.set_aspect('equal')  # Force square aspect ratio

    # Add satellite imagery if available
    try:
        ax.add_image(tiler, 6)
    except Exception as e:
        print(f"Error adding satellite imagery for {lake_name}: {e}")
        ax.add_feature(cartopy.feature.LAND, facecolor=land_color)

    # Add the highlighted study lake
    ax.add_geometries([lake.geometry], 
                      crs=ccrs.PlateCarree(), 
                      facecolor='none', 
                      edgecolor=lake_color, 
                      linewidth=2,
                      rasterized=True)

    # Set title with number and name
    number = lake_name_to_label[lake_name]
    lake_title = f"{number} {lake_name.replace('_', ' ')}"
    ax.set_title(lake_title, fontsize=16, pad=4)

# Add map tile attribution in bottom-right corner
fig.text(0.85, 0.08, "© Google Maps 2025",
         ha='right', va='bottom', fontsize=10, color='black')

# Save and show with lower DPI for faster saving
fig.savefig("./results/figures/study_site_positions.png", bbox_inches='tight', dpi=400)
# plt.show()