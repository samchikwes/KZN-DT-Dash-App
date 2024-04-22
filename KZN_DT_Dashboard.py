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

df = pd.read_excel(r'KZN C2 Custom Data CDR.xlsx')

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

#Getting min & max throughput

max_throughput = df['MeanUserDataRateKbps'].max()
min_throughput = df['MeanUserDataRateKbps'].min()

#Durban map coordinates info (for center point of plotted maps)

durban_latitude = -29.883333
durban_longitude = 31.049999

#Plot all DL failure test points

durban_map2 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

dl_fail_test_points = folium.map.FeatureGroup()

for lat, long, label in zip(dl_data_fail_df.Latitude, dl_data_fail_df.Longitude, dl_data_fail_df.LogName):
    dl_fail_test_points.add_child(
        folium.features.CircleMarker(
            [lat, long],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=label
        )
    )

durban_map2.add_child(dl_fail_test_points)

map_title5 = "KZN DL Testing Failures"
title_html5 = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{map_title5}</h1>'
durban_map2.get_root().html.add_child(folium.Element(title_html5))

durban_map2.save("map2.html")

#Plot map markers indicating DL test status (failed or non-failed)

durban_map3 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

marker_cluster = MarkerCluster()

durban_map3.add_child(marker_cluster)

for index, record in df.iterrows():
    marker = folium.Marker(
        [record['Latitude'], record['Longitude']], popup = record['LogName'],
        icon = folium.Icon(
            color = 'white', icon_color = record['Marker_Color'])
    )
    marker_cluster.add_child(marker)

map_title4 = "KZN DL Testing Status"
title_html4 = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{map_title4}</h1>'
durban_map3.get_root().html.add_child(folium.Element(title_html4))

durban_map3.save("map3.html")

# Create a dash application
app = dash.Dash(__name__)
server = app.server

# Create an app layout
app.layout = html.Div(children=[html.H1('KZN C2 DL Throughput DT Dashboard',
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

                                # TASK 2: Add a pie chart to show the total successful tests for all regions
                                # If a specific region was selected, show the Success vs. Failed proportions for the region
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Throughput range (kbps):"),
                                # TASK 3: Add a slider to select throughput
                                #dcc.RangeSlider(id='throughput-slider',...)
                                #html.Div(dcc.RangeSlider(id='throughput-slider',
                                dcc.RangeSlider(id='throughput-slider',
	                                min=0, max=550000, step=50000,
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
                                        550000: '550000'},
                                    value=[min_throughput, max_throughput]
                                ),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-throughput-scatter-chart')),
                                html.Br(),

                                #Add a scatter chart to show the correlation between throughput and SINR
                                html.Div(dcc.Graph(id='throughput-sinr-scatter-chart')),
                                html.Br(),

				#Add a bar chart for mean throughput per area
                                html.Div(dcc.Graph(id='mean-throughput-bar-chart')),
                                html.Br(),

                                #Add a bar chart for max throughput per area
                                html.Div(dcc.Graph(id='max-throughput-bar-chart')),
                                html.Br(),

                                #Add a bar chart for min throughput per area
                                html.Div(dcc.Graph(id='min-throughput-bar-chart')),
                                html.Br(),

                                #Add a violin plot for throughput per area
                                html.Div(dcc.Graph(id='throughput-violin-plot')),
                                html.Br(),

				#Add saved Folium map plot of KZN DL Failure Test Points
				html.P("Map 1 -> Zoom in and click points to view their logfile name details", style={"fontSize": 20}),
				html.Iframe(srcDoc = open('map2.html', 'r').read(), style={'width': '1050px', 'height': '510px'})
                                ])

# TASK 2:
# Add a callback function for `area-dropdown` as input, `success-pie-chart` as output
# Function decorator to specify function input and output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='area-dropdown', component_property='value'))
def get_pie_chart(entered_area):
    filtered_df = df
    if entered_area == 'ALL':
        fig = px.pie(filtered_df, values='Class',
        names='Area',
        title='KZN DL Test Samples Per Area')
        return fig
    else:
        # return the outcomes piechart for a selected region
        filtered_df=df[df['Area']== entered_area]
        filtered_df = filtered_df.groupby(['Area', 'Class']).size().reset_index(name='class count')
        fig2 = px.pie(filtered_df, values='class count',
        names='Class',
        title=f'DL Testing Status for {entered_area}')
        return fig2

# TASK 4:
# Add a callback function for `area-dropdown` and `throughput-slider` as inputs, `success-throughput-scatter-chart` as output
@app.callback(Output(component_id='success-throughput-scatter-chart', component_property='figure'),
              Input(component_id='area-dropdown', component_property='value'),
              Input(component_id="throughput-slider", component_property="value"))
def get_scatter_chart(entered_area, throughput_slider):
    filtered_df2 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    if entered_area == 'ALL':
        fig3 = px.scatter(filtered_df2, x = 'MeanUserDataRateKbps', y = 'Class', color='EndDataRadioBearer',
        title=f'Correlation Dataset: Throughput for different RATS in successful cases for All Areas')
        return fig3
    else:
        # return the outcomes scatter chart for a selected area and throughput
        filtered_df3=filtered_df2[filtered_df2['Area']== entered_area]
        fig4 = px.scatter(filtered_df3, x = 'MeanUserDataRateKbps', y = 'Class', color='EndDataRadioBearer',
        title=f'Correlation Dataset: Throughput for different RATS in successful cases for {entered_area}')
        return fig4

# Add a callback function for `area-dropdown` and `throughput-slider` as inputs, `throughput-sinr-scatter-chart` as output
@app.callback(Output(component_id='throughput-sinr-scatter-chart', component_property='figure'),
              Input(component_id='area-dropdown', component_property='value'),
              Input(component_id="throughput-slider", component_property="value"))
def get_throughput_sinr_scatter_chart(entered_area, throughput_slider):
    filtered_df4 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    if entered_area == 'ALL':
        fig5 = px.scatter(filtered_df4, x = 'SINR', y = 'MeanUserDataRateKbps',
        title='Correlation Dataset: Throughput vs SINR for All Areas')
        return fig5
    else:
        # return the scatter chart for a selected area and throughput range
        filtered_df5=filtered_df4[filtered_df4['Area']== entered_area]
        fig6 = px.scatter(filtered_df5, x = 'SINR', y = 'MeanUserDataRateKbps',
        title=f'Correlation Dataset: Throughput vs SINR for {entered_area}')
        return fig6

# Add a callback function for `mean-throughput-bar-chart` as output
@app.callback(Output(component_id='mean-throughput-bar-chart', component_property='figure'),
              Input(component_id="throughput-slider", component_property="value"))
def get_mean_bar_chart(throughput_slider):
    filtered_df6 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    df_area_throughput = filtered_df6[['Area', 'MeanUserDataRateKbps']]
    df_area_throughput2 = df_area_throughput.groupby(['Area'], as_index=False).mean()
    fig7 = px.bar(df_area_throughput2, x='Area', y='MeanUserDataRateKbps', color='Area',
                      title='KZN Average DL Throughput Per Area')
    return fig7

# Add a callback function for `max-throughput-bar-chart` as output
@app.callback(Output(component_id='max-throughput-bar-chart', component_property='figure'),
              Input(component_id="throughput-slider", component_property="value"))
def get_max_bar_chart(throughput_slider):
    filtered_df7 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    df_area_throughput3 = filtered_df7[['Area', 'MeanUserDataRateKbps']]
    df_area_throughput4 = df_area_throughput3.groupby(['Area'], as_index=False).max()
    fig8 = px.bar(df_area_throughput4, x='Area', y='MeanUserDataRateKbps', color='Area',
                      title='KZN Max DL Throughput Per Area')
    return fig8

# Add a callback function for `min-throughput-bar-chart` as output
@app.callback(Output(component_id='min-throughput-bar-chart', component_property='figure'),
              Input(component_id="throughput-slider", component_property="value"))
def get_min_bar_chart(throughput_slider):
    filtered_df8 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    df_area_throughput5 = filtered_df8[['Area', 'MeanUserDataRateKbps']]
    df_area_throughput6 = df_area_throughput5.groupby(['Area'], as_index=False).min()
    fig9 = px.bar(df_area_throughput6, x='Area', y='MeanUserDataRateKbps', color='Area',
                      title='KZN Min DL Throughput Per Area')
    return fig9

# Add a callback function for `throughput-violin-plot` as output
@app.callback(Output(component_id='throughput-violin-plot', component_property='figure'),
              Input(component_id="throughput-slider", component_property="value"))
def get_box_plot(throughput_slider):
    filtered_df10 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
    fig11 = px.violin(filtered_df10, x='Area', y='MeanUserDataRateKbps', color='Area', box=True, points="all", hover_data=filtered_df10.columns,
                      title='Throughput Violin Plot Per Area')
    return fig11

# Run the app
if __name__ == '__main__':
    app.run_server()
