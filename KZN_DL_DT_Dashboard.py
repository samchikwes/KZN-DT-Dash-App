import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
from folium.plugins import FloatImage
from folium.features import DivIcon
import branca.colormap
from collections import defaultdict
import os
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

df = pd.read_excel(r'KZN C3 Custom Data CDR.xlsx')

sites_df = pd.read_csv(r'KZN Sites 22_11_2021.csv')

#Function to classify failures and non-failures as class 0 or 1

def get_class(cl):
    if cl == 'Failed':
        marker = 0
    elif cl != 'Failed':
        marker = 1
    return marker

#Function to classify failures and non-failures as red or green color

def get_marker_color(cl):
    marker = ''
    if cl == 'Failed':
        marker = 'red'
    elif cl == 'Succeeded':
        marker = 'green'
    return marker

#Applying functions to DL testing data

df['Class'] = df['ServiceStatus'].apply(get_class)

df['Marker_Color'] = df['ServiceStatus'].apply(get_marker_color)

#Generate dataframes with DL failure and non-failure test samples

dl_data_fail_df = df[df['ServiceStatus']=='Failed']

dl_data_success_df = df[df['ServiceStatus']!='Failed']

#Function to calculate distance between two points

def dist_between_two_lat_lon(*args):
    from math import asin, cos, radians, sin, sqrt
    lat1, lat2, long1, long2 = map(radians, args)

    dist_lats = abs(lat2 - lat1)
    dist_longs = abs(long2 - long1)
    a = sin(dist_lats / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dist_longs / 2) ** 2
    c = asin(sqrt(a)) * 2
    radius_earth = 6378  # the "Earth radius" R varies from 6356.752 km at the poles to 6378.137 km at the equator.
    return c * radius_earth

#Function to get the closest point

def find_closest_lat_lon(data, v):
    try:
        return min(data, key=lambda p: dist_between_two_lat_lon(v['lat'], p['lat'], v['lon'], p['lon']))
    except TypeError:
        print('Not a list or not a number.')

#Function to get the furthest point

def find_furthest_lat_lon(data, v):
    try:
        return max(data, key=lambda p: dist_between_two_lat_lon(v['lat'], p['lat'], v['lon'], p['lon']))
    except TypeError:
        print('Not a list or not a number.')

#Function to calculate the distance between points

from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

#Code for extracting Max DL Throughput point with its coordinates

max_throughput = df['MeanUserDataRateKbps'].max()

id_max_throughput = ''

for id, rec in enumerate (df['MeanUserDataRateKbps']):
    if rec == max_throughput:
        id_max_throughput = id

identified_max_throughput_result = df.loc[id_max_throughput].at["MeanUserDataRateKbps"]
identified_max_throughput_lat = df.loc[id_max_throughput].at["Latitude"]
identified_max_throughput_lon = df.loc[id_max_throughput].at["Longitude"]

#Code for extracting Min DL Throughput point with its coordinates

min_throughput = df['MeanUserDataRateKbps'].min()

id_min_throughput = ''

for id, rec in enumerate (df['MeanUserDataRateKbps']):
    if rec == min_throughput:
        id_min_throughput = id

identified_min_throughput_result = df.loc[id_min_throughput].at["MeanUserDataRateKbps"]
identified_min_throughput_lat = df.loc[id_min_throughput].at["Latitude"]
identified_min_throughput_lon = df.loc[id_min_throughput].at["Longitude"]

#Plot all KZN sites with custom icons on a map, centred on Max DL Throughput point

durban_map12 = folium.Map(location = [identified_max_throughput_lat, identified_max_throughput_lon], zoom_start=17, control_scale=True)

#Plot Marker object with custom icon

icon_path = r"cell_tower.jpg"
icon = folium.features.CustomIcon(icon_image=icon_path ,icon_size=(50,50))

kzn_sites4 = folium.map.FeatureGroup()

for lat, long, label in zip(sites_df.Latitude, sites_df.Longitude, sites_df.ENODEB_NAME):
    kzn_sites4.add_child(
        folium.Marker(
            [lat, long],
            popup=label,
            icon=folium.features.CustomIcon(icon_image=icon_path ,icon_size=(50,50))
        )
    )

durban_map12.add_child(kzn_sites4)

#Plot Max DL Throughput point on map

max_point_coordinate = [identified_max_throughput_lat, identified_max_throughput_lon]

max_id_circle = folium.Circle(max_point_coordinate, radius=25, color='#d35400', fill=True).add_child(folium.Popup('Max DL Throughput: {s}kbps'.format(s=identified_max_throughput_result)))
durban_map12.add_child(max_id_circle)

#Represent coordinates for all sites, then find closest one to max DL throughput point

sites_check = {}

t = 0

while t < len(sites_df['ENODEB_NAME']):
    sites_check[t] = {'lat': sites_df.loc[t].at["Latitude"], 'lon': sites_df.loc[t].at["Longitude"]}
    t = t+1

site_coordinates_check4 = []

site_coordinates_check4 = [None] * len(sites_df['ENODEB_NAME'])

ao = 0

for v in sites_check.values():
    site_coordinates_check4[ao] = v
    ao+=1

max_dl_throughput_point_to_be_found = {'lat': identified_max_throughput_lat, 'lon': identified_max_throughput_lon}
nearest_site_coord_checked4 = find_closest_lat_lon(site_coordinates_check4, max_dl_throughput_point_to_be_found)

nearest_site_checked_lat4 = nearest_site_coord_checked4['lat']
nearest_site_checked_lon4 = nearest_site_coord_checked4['lon']

#Generate line to nearest site point

max_closest_site_line=folium.PolyLine(locations=([identified_max_throughput_lat, identified_max_throughput_lon], [nearest_site_checked_lat4, nearest_site_checked_lon4]), weight=3)
durban_map12.add_child(max_closest_site_line)

#Use function to find distance to closest site point

distance_to_site_checked4 = calculate_distance(identified_max_throughput_lat, identified_max_throughput_lon, nearest_site_checked_lat4, nearest_site_checked_lon4)

#Add distance marker for the line

max_site_distance_marker = folium.Marker(
   [identified_max_throughput_lat, identified_max_throughput_lon],
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "Closest site is {:10.2f} km away".format(distance_to_site_checked4)
       )
   )

durban_map12.add_child(max_site_distance_marker)

#Add title to map

max_map_title = "KZN Max DL Throughput Point"
max_title_html = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{max_map_title}</h1>'
durban_map12.get_root().html.add_child(folium.Element(max_title_html))

durban_map12.save("map12.html")

#Plot all KZN sites with custom icons on a map, centred on Min DL Throughput point

durban_map13 = folium.Map(location = [identified_min_throughput_lat, identified_min_throughput_lon], zoom_start=16, control_scale=True)

kzn_sites7 = folium.map.FeatureGroup()

for lat, long, label in zip(sites_df.Latitude, sites_df.Longitude, sites_df.ENODEB_NAME):
    kzn_sites7.add_child(
        folium.Marker(
            [lat, long],
            popup=label,
            icon=folium.features.CustomIcon(icon_image=icon_path ,icon_size=(50,50))
        )
    )

durban_map13.add_child(kzn_sites4)

#Plot Min DL Throughput point on map

min_point_coordinate = [identified_min_throughput_lat, identified_min_throughput_lon]

min_id_circle = folium.Circle(min_point_coordinate, radius=50, color='#d35400', fill=True).add_child(folium.Popup('Min DL Throughput: {s}kbps'.format(s=identified_min_throughput_result)))
durban_map13.add_child(min_id_circle)

#Represent coordinates for all sites, then find closest one to min DL throughput point

sites_check2 = {}

t2 = 0

while t2 < len(sites_df['ENODEB_NAME']):
    sites_check[t2] = {'lat': sites_df.loc[t2].at["Latitude"], 'lon': sites_df.loc[t2].at["Longitude"]}
    t2 = t2+1

site_coordinates_check5 = []

site_coordinates_check5 = [None] * len(sites_df['ENODEB_NAME'])

ao2 = 0

for v in sites_check.values():
    site_coordinates_check5[ao2] = v
    ao2+=1

min_dl_throughput_point_to_be_found = {'lat': identified_min_throughput_lat, 'lon': identified_min_throughput_lon}
nearest_site_coord_checked5 = find_closest_lat_lon(site_coordinates_check5, min_dl_throughput_point_to_be_found)

nearest_site_checked_lat5 = nearest_site_coord_checked5['lat']
nearest_site_checked_lon5 = nearest_site_coord_checked5['lon']

#Generate line to nearest site point

min_closest_site_line=folium.PolyLine(locations=([identified_min_throughput_lat, identified_min_throughput_lon], [nearest_site_checked_lat5, nearest_site_checked_lon5]), weight=3)
durban_map13.add_child(min_closest_site_line)

#Use function to find distance to closest site point

distance_to_site_checked5 = calculate_distance(identified_min_throughput_lat, identified_min_throughput_lon, nearest_site_checked_lat5, nearest_site_checked_lon5)

#Add distance marker for the line

min_site_distance_marker = folium.Marker(
   [identified_min_throughput_lat, identified_min_throughput_lon],
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "Closest site is {:10.2f} km away".format(distance_to_site_checked5)
       )
   )

durban_map13.add_child(min_site_distance_marker)

#Add title to map

min_map_title = "KZN Min DL Throughput Point"
min_title_html = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{min_map_title}</h1>'
durban_map13.get_root().html.add_child(folium.Element(min_title_html))

durban_map13.save("map13.html")

# Create a dash application
app = dash.Dash(__name__)
server = app.server

# Create an app layout
app.layout = html.Div(children=[html.H1('KZN C3 DL Testing DT Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Area selection
                                # The default select value is for ALL areas
                                # dcc.Dropdown(id='area-dropdown',...)
                                dcc.Dropdown(id='area-dropdown',
                                    options=[
                                        {'label': 'All Areas', 'value': 'ALL'},
                                        {'label': 'KZN_Lady_Smith', 'value': 'KZN_Lady_Smith'},
                                        {'label': 'KZN_Metro_DBN', 'value': 'KZN_Metro_DBN'},
                                        {'label': 'KZN_Metro_PMB', 'value': 'KZN_Metro_PMB'},
                                        {'label': 'KZN_Newcastle', 'value': 'KZN_Newcastle'},
                                        {'label': 'KZN_Richards Bay', 'value': 'KZN_Richards Bay'},
                                    ],
                                    value='ALL',
                                    placeholder="Select an Area",
                                    searchable=True
                                    ),
                                html.Br(),

				html.P("DL Throughput Range (kbps):"),
                                # TASK 3: Add a slider to select throughput
                                #dcc.RangeSlider(id='throughput-slider',...)
                                #html.Div(dcc.RangeSlider(id='throughput-slider',
                                dcc.RangeSlider(id='throughput-slider',
	                                min=0, max=750000, step=50000,
	                                marks={0: '0',
		                        50000: '50000',
		                        100000: '100000',
		                        150000: '150000',
                                        200000: '200000',
                                        250000: '250000',
                                        300000: '300000',
                                        350000: '350000',
                                        400000: '400000',
                                        450000: '450000',
                                        500000: '500000',
                                        550000: '550000',
                                        600000: '600000',
                                        650000: '650000',
                                        700000: '700000',
                                        750000: '750000'},
                                    value=[min_throughput, max_throughput]
                                ),

                                # TASK 4: Add a scatter chart to show the correlation between throughput and RAT for successful test cases
                                html.Div(dcc.Graph(id='success-throughput-scatter-chart')),
                                html.Br(),

                                #Add a box plot for throughput per area
                                html.Div(dcc.Graph(id='throughput-box-plot')),
                                html.Br(),

                                #Add a box plot for throughput per tech
                                html.Div(dcc.Graph(id='throughput-box-plot-2')),
                                html.Br(),

				# Add saved Folium map plot of KZN Max DL Throughput Point
                                html.P("Map 1 -> Click on max DL throughput test point and site icons for further details", style={"fontSize": 20}),
                                html.Iframe(srcDoc = open('map12.html', 'r').read(), style={'width': '1050px', 'height': '510px'}),
                                html.Br(),

                                # Add saved Folium map plot of KZN Min DL Throughput Point
                                html.P("Map 2 -> Click on min DL throughput point and site icons for further details", style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map13.html', 'r').read(), style={'width': '1050px', 'height': '510px'})
                                ])

# TASK 4:
# Add a callback function for `area-dropdown` and `throughput-slider` as inputs, `success-throughput-scatter-chart` as output
@app.callback(Output(component_id='success-throughput-scatter-chart', component_property='figure'),
              Input(component_id='area-dropdown', component_property='value'),
              Input(component_id="throughput-slider", component_property="value"))
def get_scatter_chart(entered_area, throughput_slider):
    filtered_df2 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    if entered_area == 'ALL':
        fig3 = px.scatter(filtered_df2, x = 'MeanUserDataRateKbps', y = 'Class', color='EndDataRadioBearer',
        title=f'Correlation Dataset: KZN DL Throughput for Different Techs in Successful Test Cases for All Areas')
        return fig3
    else:
        # return the outcomes scatter chart for a selected area and throughput
        filtered_df3=filtered_df2[filtered_df2['Area']== entered_area]
        fig4 = px.scatter(filtered_df3, x = 'MeanUserDataRateKbps', y = 'Class', color='EndDataRadioBearer',
        title=f'Correlation Dataset: DL Throughput for Different Techs in Successful Cases for {entered_area}')
        return fig4

# Add a callback function for `throughput-box-plot` as output
@app.callback(Output(component_id='throughput-box-plot', component_property='figure'),
              Input(component_id="throughput-slider", component_property="value"))
def get_box_plot(throughput_slider):
    filtered_df10 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    fig11 = px.box(filtered_df10, x='Area', y='MeanUserDataRateKbps', color='Area',
                      title='KZN DL Throughput Boxplot Per Area')
    return fig11

# Add a callback function for `throughput-box-plot-2` as output
@app.callback(Output(component_id='throughput-box-plot-2', component_property='figure'),
              Input(component_id="throughput-slider", component_property="value"))
def get_box_plot(throughput_slider):
    filtered_df11 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    fig12 = px.box(filtered_df11, x='EndDataRadioBearer', y='MeanUserDataRateKbps', color='EndDataRadioBearer',
                      title='KZN DL Throughput Boxplot Per Tech')
    return fig12

# Run the app
if __name__ == '__main__':
    app.run_server()
