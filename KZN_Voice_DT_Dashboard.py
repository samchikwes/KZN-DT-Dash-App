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

df = pd.read_excel(r'KZN C3 Customized Voice CDR CQI.xlsx')

df2 = pd.read_excel(r'KZN C3 MOS samples.xlsx')

#Replace commas in coordinate values with dots
df["Latitude"] = df["Latitude"].str.replace(',', '.')
df["Longitude"] = df["Longitude"].str.replace(',', '.')

df2["Latitude"] = df2["Latitude"].str.replace(',', '.')
df2["Longitude"] = df2["Longitude"].str.replace(',', '.')

#Replace commas in MOS values with dots
df2["POLQA Value Overall"] = df2["POLQA Value Overall"].str.replace(',', '.')

#Replace commas in SINR values with dots
df2["Mo Start SINR"] = df2["Mo Start SINR"].str.replace(',', '.')
df2["Mt Start SINR"] = df2["Mt Start SINR"].str.replace(',', '.')

#Generate scatter plot of Throughput and SINR

df_mos_throughput = df2
df_mos_throughput = df_mos_throughput.dropna(subset=['POLQA Value Overall', 'Mo Start SINR'])

df_mos_throughput['Mo Start SINR'] = df_mos_throughput['Mo Start SINR'].astype(float)
df_mos_throughput['Mo Start SINR'] = df_mos_throughput['Mo Start SINR'].astype(int)
df_mos_throughput['POLQA Value Overall'] = df_mos_throughput['POLQA Value Overall'].astype(float)

#Generate dataframes with voice failure samples

voice_drop_df = df[df['CallDropped']== 1]
voice_setup_fail_df = df[df['CallSetup']== 0]
voice_unsustain_df = df[df['CallNonSustain']== 1]

#Durban map coordinates info (for center point of plotted maps)

durban_latitude = -29.883333
durban_longitude = 31.049999

#Plot all voice failure test points

durban_map2 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

voice_drop_test_points = folium.map.FeatureGroup()

for lat, long, label in zip(voice_drop_df.Latitude, voice_drop_df.Longitude, voice_drop_df.LogName):
    voice_drop_test_points.add_child(
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

durban_map2.add_child(voice_drop_test_points)

voice_setup_fail_test_points = folium.map.FeatureGroup()

for lat, long, label in zip(voice_setup_fail_df.Latitude, voice_setup_fail_df.Longitude, voice_setup_fail_df.LogName):
    voice_setup_fail_test_points.add_child(
        folium.features.CircleMarker(
            [lat, long],
            radius=3,
            color='brown',
            fill=True,
            fill_color='brown',
            fill_opacity=0.6,
            popup=label
        )
    )

durban_map2.add_child(voice_setup_fail_test_points)

voice_unsustain_test_points = folium.map.FeatureGroup()

for lat, long, label in zip(voice_unsustain_df.Latitude, voice_unsustain_df.Longitude, voice_unsustain_df.LogName):
    voice_unsustain_test_points.add_child(
        folium.features.CircleMarker(
            [lat, long],
            radius=3,
            color='orange',
            fill=True,
            fill_color='orange',
            fill_opacity=0.6,
            popup=label
        )
    )

durban_map2.add_child(voice_unsustain_test_points)

item_txt = """<br> &nbsp; {item} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col}"></i>"""
item_txt2 = """<br> &nbsp; {item2} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col2}"></i>"""
item_txt3 = """<br> &nbsp; {item3} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col3}"></i>"""
html_itms = item_txt.format(item="Call Drop", col="red")
html_itms2 = item_txt2.format(item2="Call Setup Failure", col2="brown")
html_itms3 = item_txt3.format(item3="Unsustainable Call", col3="orange")

legend_html = """
     <div style="
     position: fixed; 
     bottom: 40px; left: 5px; width: 180px; height: 85px; 
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

      </div> """.format(title="Legend", itm_txt=html_itms, itm_txt2=html_itms2, itm_txt3=html_itms3)
durban_map2.get_root().html.add_child(folium.Element(legend_html))

map_title5 = "KZN Voice Unqualified Samples"
title_html5 = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{map_title5}</h1>'
durban_map2.get_root().html.add_child(folium.Element(title_html5))

durban_map2.save("map2.html")

#Generate heatmap of DL Throughput with colour scale

map_osm = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

steps=20
colormap = branca.colormap.linear.YlOrRd_09.scale(0, 1).to_step(steps)
gradient_map=defaultdict(dict)
for i in range(steps):
    gradient_map[1/steps*i] = colormap.rgb_hex_str(1/steps*i)
colormap.add_to(map_osm) #add color bar at the top of the map

df_non_null_loc = df_mos_throughput
df_non_null_loc = df_non_null_loc.dropna(subset=['Latitude', 'Longitude', 'POLQA Value Overall'])

mos_data = [[row['Latitude'],row['Longitude'], row['POLQA Value Overall']] for index, row in df_non_null_loc.iterrows()]
HeatMap(mos_data, gradient = gradient_map).add_to(map_osm) # Add heat map to the previously created map

map_title3 = "KZN MOS Heatmap"
title_html3 = f'<h1 style="position:absolute;z-index:100000;left:20vw" >{map_title3}</h1>'
map_osm.get_root().html.add_child(folium.Element(title_html3))

map_osm.save("map5.html")

# Create a dash application
app = dash.Dash(__name__)
server = app.server

# Create an app layout
app.layout = html.Div(children=[html.H1('KZN C3 Voice Testing DT Dashboard',
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

                                # TASK 3: Add a slider to select MOS
                                html.P("MOS Range:"),
                                #dcc.RangeSlider(id='MOS-slider',...)
                                #html.Div(dcc.RangeSlider(id='MOS-slider',
                                dcc.RangeSlider(id='mos-slider',
	                                min=0, max=5, step=0.5,
	                                marks={0: '0',
		                                0.5: '0.5',
		                                1: '1',
		                                1.5: '1.5',
                                        2: '2',
                                        2.5: '2.5',
                                        3: '3',
                                        3.5: '3.5',
                                        4: '4',
                                        4.5: '4.5',
                                        5: '5'},
                                    value=[0, 5]
                                ),

                                #Add a scatter chart to show the correlation between MOS and SINR
                                html.Div(dcc.Graph(id='mos-sinr-scatter-chart')),
                                html.Br(),

                                # Add a bar chart for mean throughput per area
                                html.Div(dcc.Graph(id='mean-mos-bar-chart')),
                                html.Br(),

                                # Add a bar chart for max throughput per area
                                html.Div(dcc.Graph(id='max-mos-bar-chart')),
                                html.Br(),

                                # Add a bar chart for min throughput per area
                                html.Div(dcc.Graph(id='min-mos-bar-chart')),
                                html.Br(),

                                # Add saved Folium map plot of KZN Voice Failure Test Points
                                html.P("Map 1 -> Zoom in and click unqualified sample points to view their logfile name details",
                                       style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map2.html', 'r').read(),
                                            style={'width': '1050px', 'height': '510px'}),

                                html.Br(),

                                # Add saved Folium map plot of KZN MOS Heatmap
                                html.P("Map 2 -> Zoom in for better granularity view of areas, with darker shading indicating proportionally higher MOS areas",
                                        style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map5.html', 'r').read(),
                                            style={'width': '1050px', 'height': '510px'})
                                ])

# Add a callback function for `area-dropdown` and `mos-slider` as inputs, `mos-sinr-scatter-chart` as output
@app.callback(Output(component_id='mos-sinr-scatter-chart', component_property='figure'),
              Input(component_id='area-dropdown', component_property='value'),
              Input(component_id="mos-slider", component_property="value"))
def get_mos_sinr_scatter_chart(entered_area, mos_slider):
    filtered_df4 = df_mos_throughput[(df_mos_throughput['POLQA Value Overall']>=mos_slider[0]) & (df_mos_throughput['POLQA Value Overall']<=mos_slider[1])]
    if entered_area == 'ALL':
        fig5 = px.scatter(filtered_df4, x = 'Mo Start SINR', y = 'POLQA Value Overall',
        title='Correlation Dataset: Mobile Originating MOS vs SINR for All Areas')
        return fig5
    else:
        # return the scatter chart for a selected area and throughput range
        filtered_df5=filtered_df4[filtered_df4['Area']== entered_area]
        fig6 = px.scatter(filtered_df5, x = 'Mo Start SINR', y = 'POLQA Value Overall',
        title=f'Correlation Dataset: Mobile Originating MOS vs SINR for {entered_area}')
        return fig6

# Add a callback function for `mean-mos-bar-chart` as output
@app.callback(Output(component_id='mean-mos-bar-chart', component_property='figure'),
              Input(component_id="mos-slider", component_property="value"))
def get_mean_bar_chart(mos_slider):
    filtered_df6 = df_mos_throughput[(df_mos_throughput['POLQA Value Overall']>=mos_slider[0]) & (df_mos_throughput['POLQA Value Overall']<=mos_slider[1])]
    df_area_mos = filtered_df6[['Area', 'POLQA Value Overall']]
    df_area_mos2 = df_area_mos.groupby(['Area'], as_index=False).mean()
    fig7 = px.bar(df_area_mos2, x='Area', y='POLQA Value Overall', color='Area',
                      title='KZN Average MOS Per Area')
    return fig7

# Add a callback function for `max-mos-bar-chart` as output
@app.callback(Output(component_id='max-mos-bar-chart', component_property='figure'),
              Input(component_id="mos-slider", component_property="value"))
def get_max_bar_chart(mos_slider):
    filtered_df8 = df_mos_throughput[(df_mos_throughput['POLQA Value Overall']>=mos_slider[0]) & (df_mos_throughput['POLQA Value Overall']<=mos_slider[1])]
    df_area_mos3 = filtered_df8[['Area', 'POLQA Value Overall']]
    df_area_mos4 = df_area_mos3.groupby(['Area'], as_index=False).max()
    fig8 = px.bar(df_area_mos4, x='Area', y='POLQA Value Overall', color='Area',
                      title='KZN Max MOS Per Area')
    return fig8

# Add a callback function for `min-mos-bar-chart` as output
@app.callback(Output(component_id='min-mos-bar-chart', component_property='figure'),
              Input(component_id="mos-slider", component_property="value"))
def get_min_bar_chart(mos_slider):
    filtered_df9 = df_mos_throughput[(df_mos_throughput['POLQA Value Overall']>=mos_slider[0]) & (df_mos_throughput['POLQA Value Overall']<=mos_slider[1])]
    df_area_mos5 = filtered_df9[['Area', 'POLQA Value Overall']]
    df_area_mos6 = df_area_mos5.groupby(['Area'], as_index=False).min()
    fig9 = px.bar(df_area_mos6, x='Area', y='POLQA Value Overall', color='Area',
                      title='KZN Min MOS Per Area')
    return fig9

# Run the app
if __name__ == '__main__':
    app.run_server()
