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

df = pd.read_excel(r'C:\Temp\ZA DT Dashboard\KZN C3 Custom Streaming CDR.xlsx')

#Replace commas in coordinate values with dots
df["Latitude"] = df["Latitude"].str.replace(',', '.')
df["Longitude"] = df["Longitude"].str.replace(',', '.')

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

#Applying functions to YT testing data

df['Class'] = df['ServiceStatus'].apply(get_class)

df['Marker_Color'] = df['ServiceStatus'].apply(get_marker_color)

#Generate dataframes with YT failure and non-failure test samples

yt_fail_df = df[df['ServiceStatus']=='Failed']

yt_success_df = df[df['ServiceStatus']!='Failed']

#Durban map coordinates info (for center point of plotted maps)

durban_latitude = -29.883333
durban_longitude = 31.049999

#Plot RAT for all test points

def get_RAT_color(rt):
    marker = ''
    if rt == 'EGPRS':
        marker = 'red'
    elif rt == 'HSDPA':
        marker = 'orange'
    elif rt == 'HSPA':
        marker = 'yellow'
    elif rt == 'LTE':
        marker = 'green'
    elif rt == 'Mixed':
        marker = 'blue'
    elif rt == 'Mixed HSPA':
        marker = 'indigo'
    elif rt == 'Mixed (LTE-NR)':
        marker = 'violet'
    elif rt == 'R99':
        marker = 'grey'
    return marker

yt_RAT_df = df[['Latitude', 'Longitude', 'LogName', 'EndDataRadioBearer']]
yt_RAT_df = yt_RAT_df.dropna(subset=['Latitude', 'Longitude', 'LogName', 'EndDataRadioBearer'])

yt_RAT_df['Marker_Colour'] = yt_RAT_df['EndDataRadioBearer'].apply(get_RAT_color)

durban_map6 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

yt_RAT_points = folium.map.FeatureGroup()

for lat, long, colour, label in zip(yt_RAT_df.Latitude, yt_RAT_df.Longitude, yt_RAT_df.Marker_Colour, yt_RAT_df.LogName):
    yt_RAT_points.add_child(
        folium.features.CircleMarker(
            [lat, long],
            radius=3,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.6,
            popup=label
        )
    )

durban_map6.add_child(yt_RAT_points)

item_txt = """<br> &nbsp; {item} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col}"></i>"""
item_txt2 = """<br> &nbsp; {item2} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col2}"></i>"""
item_txt3 = """<br> &nbsp; {item3} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col3}"></i>"""
item_txt4 = """<br> &nbsp; {item4} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col4}"></i>"""
item_txt5 = """<br> &nbsp; {item5} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col5}"></i>"""
item_txt6 = """<br> &nbsp; {item6} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col6}"></i>"""
item_txt7 = """<br> &nbsp; {item7} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col7}"></i>"""
item_txt8 = """<br> &nbsp; {item8} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col8}"></i>"""
html_itms = item_txt.format(item="EGPRS", col="red")
html_itms2 = item_txt2.format(item2="HSDPA", col2="orange")
html_itms3 = item_txt3.format(item3="HSPA", col3="yellow")
html_itms4 = item_txt4.format(item4="LTE", col4="green")
html_itms5 = item_txt5.format(item5="Mixed", col5="blue")
html_itms6 = item_txt6.format(item6="Mixed HSPA", col6="indigo")
html_itms7 = item_txt7.format(item7="Mixed (LTE-NR)", col7="violet")
html_itms8 = item_txt8.format(item8="R99", col8="grey")

legend_html = """
     <div style="
     position: fixed; 
     bottom: 40px; left: 5px; width: 160px; height: 190px; 
     border:2px solid grey; z-index:9999; 

     background-color:white;
     opacity: .85;

     font-size:14px;
     font-weight: bold;

     ">
     &nbsp; {title} 

     {itm_txt}
     {itm_txt2}
     {itm_txt3}
     {itm_txt4}
     {itm_txt5}
     {itm_txt6}
     {itm_txt7}
     {itm_txt8}

      </div> """.format(title="Legend", itm_txt=html_itms, itm_txt2=html_itms2, itm_txt3=html_itms3, itm_txt4=html_itms4, itm_txt5=html_itms5, itm_txt6=html_itms6, itm_txt7=html_itms7, itm_txt8=html_itms8)
durban_map6.get_root().html.add_child(folium.Element(legend_html))

map_title2 = "KZN YT Testing Radio Access Technologies"
title_html2 = f'<h1 style="position:absolute;z-index:100000;left:20vw" >{map_title2}</h1>'
durban_map6.get_root().html.add_child(folium.Element(title_html2))

durban_map6.save("map6.html")

# Create a dash application
app = dash.Dash(__name__)
server = app.server

# Create an app layout
app.layout = html.Div(children=[html.H1('KZN C3 YT Testing DT Dashboard',
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

                                # Add saved Folium map plot of KZN DL Radio Access Technologies
                                html.P(
                                    "Map 1 -> Points indicate serving technology of the sample - zoom in and use legend to determine the tech",
                                    style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map6.html', 'r').read(),
                                            style={'width': '1050px', 'height': '510px'})
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
        title='Proportion of KZN YT Test Samples Per Area')
        return fig
    else:
        # return the outcomes piechart for a selected region
        filtered_df=df[df['Area']== entered_area]
        filtered_df = filtered_df.groupby(['Area', 'Class']).size().reset_index(name='class count')
        fig2 = px.pie(filtered_df, values='class count',
        names='Class',
        title=f'YT Testing Status for {entered_area}')
        return fig2

# Run the app
if __name__ == '__main__':
    app.run_server()
