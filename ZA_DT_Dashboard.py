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
import webbrowser
import os
import dash
#import dash_html_components as html
from dash import html
#import dash_core_components as dcc
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',8)

df = pd.read_excel(r'KZN C2 Custom Data CDR.xlsx')

sites_df = pd.read_csv(r'KZN Sites 22_11_2021.csv')

#EDA

a = df.head()
print(a)
print('')

b = df.shape
print(b)
print('')

missing_data = df.isnull()
c = missing_data.head()
print(c)
print('')

d = missing_data.shape
print(d)
print('')

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

print('')

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
e = df
print(e)
print('')

df['Marker_Color'] = df['ServiceStatus'].apply(get_marker_color)
f = df
print(f)
print('')

#Generate dataframes with DL failure and non-failure test samples

dl_data_fail_df = df[df['ServiceStatus']=='Failed']
print(dl_data_fail_df)
print('')

dl_data_success_df = df[df['ServiceStatus']!='Failed']

#Generate max, mean and min DL Throughput

max_throughput = df['MeanUserDataRateKbps'].max()
mean_throughput = df['MeanUserDataRateKbps'].mean()
min_throughput = df['MeanUserDataRateKbps'].min()
print(max_throughput)
print(mean_throughput)
print(min_throughput)
print('')

#dl_count_df = df[df['ServiceStatus']]
#dl_success_count_df = dl_data_success_df[dl_data_success_df['ServiceStatus']]
#dl_fail_count_df = dl_data_fail_df[dl_data_fail_df['ServiceStatus']]

#dl_success_rate = (dl_data_success_count_df.count()/dl_count_df.count())*100
#dl_success_rate = (dl_data_success_df.count()/df.count())*100
dl_success_rate = (dl_data_success_df['ServiceStatus'].count()/df['ServiceStatus'].count())*100
print(dl_success_rate)
print('')

#DF of 'Area' and 'DL Throughout'

df_area_throughput = df[['Area', 'MeanUserDataRateKbps']]
g = df_area_throughput.head()
print(g)
print('')

#Display average throughput per KZN test area

df_area_throughput = df_area_throughput.groupby(['Area'], as_index=False).mean()
h = df_area_throughput
print(h)
print('')

#Generate scatter plot of Throughput and SINR

df_sinr_throughput = df
df_sinr_throughput = df_sinr_throughput.dropna(subset=['MeanUserDataRateKbps', 'SINR'])

sns.catplot(y='MeanUserDataRateKbps', x='SINR', data=df_sinr_throughput)
plt.xlabel("SINR", fontdict={"weight": "bold"})
plt.ylabel("MeanUserDataRateKbps", fontdict={"weight": "bold"})
plt.show()

#Generate scatter plot of Throughput and SINR (binned)

bins = [-20, -10, 0, 10, 20, 30, 40]
df_sinr_throughput['SINR_Binned'] = pd.cut(df_sinr_throughput['SINR'], bins)

#sns.catplot(y='MeanUserDataRateKbps', x='SINR', data=df_sinr_throughput)
sns.catplot(y='MeanUserDataRateKbps', x='SINR_Binned', data=df_sinr_throughput)
plt.xlabel("SINR", fontdict={"weight": "bold"})
plt.ylabel("MeanUserDataRateKbps", fontdict={"weight": "bold"})
plt.show()

#Generate scatter plot of Throughput and SINR (integer)

df_sinr_throughput['SINR'] = df_sinr_throughput['SINR'].astype(int)

hi = sns.catplot(y='MeanUserDataRateKbps', x='SINR', data=df_sinr_throughput)
hi.fig.suptitle('KZN DL Throughput vs SINR',
                fontsize=24, fontdict={"weight": "bold"})
plt.xlabel("SINR", fontdict={"weight": "bold"})
#plt.ylabel("MeanUserDataRateKbps", fontdict={"weight": "bold"})
plt.ylabel("MeanUserDataRateKbps", fontdict={"weight": "bold"})
plt.show()

#Generate barlot of average throughput per KZN test area

ax = sns.barplot(y='MeanUserDataRateKbps', x='Area', data=df_area_throughput)
ax.set_title('KZN Mean DL Throughput Per Area',
             fontdict={'fontsize': 20,
                        'fontweight': 'bold',
                        'color': 'black'})
ax.bar_label(ax.containers[0])
#sns.barplot(y='MeanUserDataRateKbps', x='Area', data=df_area_throughput).set_title('KZN Mean DL Throughput Per Area',
#                                                                                     fontdict={'fontsize': 20,
#                                                                                               'fontweight': 'bold',
#                                                                                               'color': 'black'})
plt.xlabel('Area', fontdict={"weight": "bold"})
plt.ylabel('MeanUserDataRateKbps', fontdict={"weight": "bold"})
plt.show()

#Generate max throughput per KZN test area

df_area_max_throughput = df[['Area', 'MeanUserDataRateKbps']]

df_area_max_throughput = df_area_max_throughput.groupby(['Area'], as_index=False).max()

#Generate barlot of max throughput per KZN test area

ax2 = sns.barplot(y='MeanUserDataRateKbps', x='Area', data=df_area_max_throughput)
ax2.set_title('KZN Max DL Throughput Per Area',
             fontdict={'fontsize': 20,
                        'fontweight': 'bold',
                        'color': 'black'})
ax2.bar_label(ax2.containers[0])
#sns.barplot(y='MeanUserDataRateKbps', x='Area', data=df_area_max_throughput).set_title('KZN Max DL Throughput Per Area',
#                                                                                     fontdict={'fontsize': 20,
#                                                                                               'fontweight': 'bold',
#                                                                                               'color': 'black'})
plt.xlabel('Area', fontdict={"weight": "bold"})
plt.ylabel('MeanUserDataRateKbps', fontdict={"weight": "bold"})
plt.show()

#Group by area and throughput

i = df[['Area', 'MeanUserDataRateKbps']]
print(i)

df_i = i.groupby('Area', axis = 0).sum()
print(df_i)

#Group by area and class

j = df[['Area', 'Class']]
print(j)

df_j = j.groupby('Area', axis = 0).sum()
print(df_j)

#Group by class and serving MNC

k = df[['Class', 'ServingMNC']]
print(k)

df_k = k.groupby('Class', axis = 0).sum()
print(df_k)

#Group by area and serving MNC

l = df[['Area', 'ServingMNC']]
print(l)

df_l = l.groupby('Area', axis = 0).sum()
print(df_l)

#Generate pie chart of DL testing success

#df['class'].plot(kind='pie')
#plt.title('Pie Chart of DL Testing Success')
#plt.show()

#Generate pie chart of throughput per area

df_i['MeanUserDataRateKbps'].plot(kind='pie',
#                                               figsize=(5, 6),
                                               autopct='%1.1f%%',
                                               startangle=90,
#                                               shadow=True,
                                               labels=None,
                                               pctdistance=1.12
                                               )
plt.title('Pie Chart of Area Throughput', y=1.08,
          fontdict={'fontsize': 20,
                   'fontweight': 'bold',
                    'color': 'black'})
plt.axis('equal')
plt.legend(labels = df_i.index, loc = 'upper left')
plt.show()

#Generate pie chart of ratio of test samples per area

df_l['ServingMNC'].plot(kind='pie',
#                       figsize=(5, 6),
                        autopct='%1.1f%%',
                        startangle=90,
#                       shadow=True,
                        labels=None,
                        pctdistance=1.12
                        )
plt.title('KZN DL Test Samples Per Area', y=1.08,
          fontdict={'fontsize': 20,
                    'fontweight': 'bold',
                    'color': 'black'})
plt.axis('equal')
plt.legend(labels = df_i.index, loc = 'upper left')
plt.ylabel('Test Samples', fontdict={"weight": "bold"})
plt.show()

#Generate pie chart of Dl testing success class

df_k['ServingMNC'].plot(kind='pie',
#                       figsize=(5, 6),
                        autopct='%1.1f%%',
                        startangle=90,
#                       shadow=True,
                        labels=None,
                        pctdistance=1.12
                        )
plt.title('KZN DL Testing Status', y=1.08,
          fontdict={'fontsize': 20,
                    'fontweight': 'bold',
                    'color': 'black'})
plt.axis('equal')
plt.legend(labels = df_k.index, loc = 'upper left')
plt.ylabel('Success Class', fontdict={"weight": "bold"})
plt.show()

#Durban map coordinates info (for center point of plotted maps)

durban_latitude = -29.883333
durban_longitude = 31.049999

#Plot all DL Test points

durban_map = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

dl_test_points = folium.map.FeatureGroup()

for lat, long, label in zip(df.Latitude, df.Longitude, df.LogName):
    dl_test_points.add_child(
        folium.features.CircleMarker(
            [lat, long],
            radius=5,
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=label
        )
    )

durban_map.add_child(dl_test_points)

durban_map.save("map1.html")
webbrowser.open("map1.html")

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

map_title5 = "KZN DL Testing Failure Points"
title_html5 = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{map_title5}</h1>'
durban_map2.get_root().html.add_child(folium.Element(title_html5))

W, H = (300,200)
im5 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im5)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg5 = "CLICK POINTS TO VIEW THEIR LOGFILE NAME DETAILS!"
w, h = draw.textsize(msg5)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg5,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im5.crop((0, 0,2*w,2*h)).save("pycoatextlogo5.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo5.png", bottom=10, left=0).add_to(durban_map2)

durban_map2.save("map2.html")
webbrowser.open("map2.html")

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

map_title4 = "KZN DL Testing Status Markers"
title_html4 = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{map_title4}</h1>'
durban_map3.get_root().html.add_child(folium.Element(title_html4))

W, H = (300,200)
im4 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im4)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg4 = "CLICK MARKER CLUSTERS TO FOCUS ON SPECIFIC AREAS!"
w, h = draw.textsize(msg4)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg4,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im4.crop((0, 0,2*w,2*h)).save("pycoatextlogo4.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo4.png", bottom=10, left=0).add_to(durban_map3)

W, H = (300,200)
im13 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im13)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg13 = "GREEN MARKER IS SUCCESSFUL SAMPLE AND RED FAILURE!"
w, h = draw.textsize(msg13)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg13,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im13.crop((0, 0,2*w,2*h)).save("pycoatextlogo13.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo13.png", bottom=7.5, left=0).add_to(durban_map3)

durban_map3.save("map3.html")
webbrowser.open("map3.html")

#Generate heatmap of DL Throughput

df_non_null_loc = df
df_non_null_loc = df_non_null_loc.dropna(subset=['Latitude', 'Longitude', 'MeanUserDataRateKbps'])
print(df_non_null_loc)

durban_map4 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)
dl_throughput_data = [[row['Latitude'],row['Longitude'], row['MeanUserDataRateKbps']] for index, row in df_non_null_loc.iterrows()]
HeatMap(dl_throughput_data).add_to(durban_map4)

durban_map4.save("map4.html")
webbrowser.open("map4.html")

#Generate heatmap of DL Throughput with colour scale

map_osm = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

steps=20
#colormap = branca.colormap.linear.RdBu_09.scale(0, 600000).to_step(steps)
#colormap = branca.colormap.linear.RdYlGn_09.scale(0, 1).to_step(steps)
colormap = branca.colormap.linear.YlOrRd_09.scale(0, 1).to_step(steps)
#colormap = branca.colormap.linear.YlOrRd_09.scale(0, 600000).to_step(steps)
#colormap = branca.colormap.linear.PuRd_09.scale(0, 1).to_step(steps)
gradient_map=defaultdict(dict)
for i in range(steps):
#    gradient_map[1/steps*i] = colormap.rgb_hex_str(600000/steps*i)
    gradient_map[1/steps*i] = colormap.rgb_hex_str(1/steps*i)
colormap.add_to(map_osm) #add color bar at the top of the map

HeatMap(dl_throughput_data, gradient = gradient_map).add_to(map_osm) # Add heat map to the previously created map

map_title3 = "KZN DL Throughput Heatmap"
title_html3 = f'<h1 style="position:absolute;z-index:100000;left:20vw" >{map_title3}</h1>'
map_osm.get_root().html.add_child(folium.Element(title_html3))

#Add text overlay to map

W, H = (300,200)
im2 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im2)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg2 = "ZOOM IN FOR BETTER GRANULARITY VIEW OF AREAS!"
w, h = draw.textsize(msg2)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg2,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im2.crop((0, 0,2*w,2*h)).save("pycoatextlogo2.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo2.png", bottom=10, left=0).add_to(map_osm)

W, H = (300,200)
im3 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im3)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg3 = "DARKER SHADING INDICATES HIGHER THROUGHPUT AREAS!"
w, h = draw.textsize(msg3)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg3,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im3.crop((0, 0,2*w,2*h)).save("pycoatextlogo3.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo3.png", bottom=7.5, left=0).add_to(map_osm)

map_osm.save("map5.html")
webbrowser.open("map5.html")

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

dl_RAT_df = df[['Latitude', 'Longitude', 'LogName', 'EndDataRadioBearer']]
dl_RAT_df = dl_RAT_df.dropna(subset=['Latitude', 'Longitude', 'LogName', 'EndDataRadioBearer'])

dl_RAT_df['Marker_Colour'] = dl_RAT_df['EndDataRadioBearer'].apply(get_RAT_color)

durban_map6 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

dl_RAT_points = folium.map.FeatureGroup()

for lat, long, colour, label in zip(dl_RAT_df.Latitude, dl_RAT_df.Longitude, dl_RAT_df.Marker_Colour, dl_RAT_df.LogName):
    dl_RAT_points.add_child(
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

durban_map6.add_child(dl_RAT_points)
#durban_map6.add_child(folium.map.LayerControl())

item_txt = """<br> &nbsp; {item} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col}"></i>"""
item_txt2 = """<br> &nbsp; {item2} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col2}"></i>"""
item_txt3 = """<br> &nbsp; {item3} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col3}"></i>"""
item_txt4 = """<br> &nbsp; {item4} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col4}"></i>"""
item_txt5 = """<br> &nbsp; {item5} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col5}"></i>"""
item_txt6 = """<br> &nbsp; {item6} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col6}"></i>"""
item_txt7 = """<br> &nbsp; {item7} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col7}"></i>"""
item_txt8 = """<br> &nbsp; {item8} &nbsp; <i class="fa fa-map-marker fa-1x" style="color:{col8}"></i>"""
#html_itms = item_txt.format(item="mark_1", col="red")
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
     bottom: 30px; left: 50px; width: 200px; height: 180px; 
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

map_title2 = "KZN DL Testing Radio Access Technologies"
title_html2 = f'<h1 style="position:absolute;z-index:100000;left:20vw" >{map_title2}</h1>'
durban_map6.get_root().html.add_child(folium.Element(title_html2))

durban_map6.save("map6.html")
webbrowser.open("map6.html")

#Plot all KZN sites on map as Circlemarker objects

durban_map7 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

kzn_sites = folium.map.FeatureGroup()

for lat, long, label in zip(sites_df.Latitude, sites_df.Longitude, sites_df.ENODEB_NAME):
    kzn_sites.add_child(
        folium.features.CircleMarker(
            [lat, long],
            radius=2,
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=0.6,
            popup=label
        )
    )

durban_map7.add_child(kzn_sites)

durban_map7.save("map7.html")
webbrowser.open("map7.html")

#Plot all KZN sites on map as Marker objects

durban_map8 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

kzn_sites2 = folium.map.FeatureGroup()

for lat, long, label in zip(sites_df.Latitude, sites_df.Longitude, sites_df.ENODEB_NAME):
    kzn_sites2.add_child(
        folium.Marker(
            [lat, long],
            popup=label,
            icon = folium.Icon(color = 'blue', icon='tower')
        )
    )

durban_map8.add_child(kzn_sites2)

durban_map8.save("map8.html")
webbrowser.open("map8.html")

#Plot Marker object with custom icon

durban_map_test= folium.Map(location = [durban_latitude, durban_longitude], zoom_start=15, control_scale=True)

encoded = base64.b64encode(open('C:\Temp\ZA DT Dashboard\cell_tower.jpg', 'rb').read())
decoded = base64.b64decode(encoded)
icon_url = BytesIO(decoded)
#icon = folium.features.CustomIcon(icon_url, icon_size=(50,50))
icon_path = r"C:\Temp\ZA DT Dashboard\cell_tower.jpg"
icon = folium.features.CustomIcon(icon_image=icon_path ,icon_size=(50,50))
folium.Marker([-29.883333, 31.049999],
              popup='Cell Tower',
              icon=icon
              ).add_to(durban_map_test)

durban_map_test.save("maptest.html")
webbrowser.open("maptest.html")

#Plot all KZN sites on map as Marker objects with custom icon

durban_map_test2 = folium.Map(location = [durban_latitude, durban_longitude], zoom_start=7, control_scale=True)

kzn_sites3 = folium.map.FeatureGroup()

for lat, long, label in zip(sites_df.Latitude, sites_df.Longitude, sites_df.ENODEB_NAME):
    kzn_sites3.add_child(
        folium.Marker(
            [lat, long],
            popup=label,
            icon=folium.features.CustomIcon(icon_image=icon_path ,icon_size=(50,50))
        )
    )

durban_map_test2.add_child(kzn_sites3)

durban_map_test2.save("maptest2.html")
webbrowser.open("maptest2.html")

#Zooming in to Durban point on map with sites added to map

kzn_map= folium.Map(location = [durban_latitude, durban_longitude], zoom_start=12, control_scale=True)

kzn_sites = folium.map.FeatureGroup()

kzn_map.add_child(kzn_sites)

durban_coordinate = [durban_latitude, durban_longitude]

circle = folium.Circle(durban_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('Durban Point'))
kzn_map.add_child(circle)

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

#Practice generating nearest and furthest point

RAN_Marine_Dr_North_KZN = {'lat': -29.9006472222222, 'lon': 31.0381}
RAN_Ansteys_Beach_KZN = {'lat': -29.912917, 'lon': 31.029139}
RAN_Brighton_Beach_East_KZN = {'lat': -29.929976, 'lon': 31.010907}

site_list = [RAN_Marine_Dr_North_KZN, RAN_Ansteys_Beach_KZN, RAN_Brighton_Beach_East_KZN]

point_to_find = {'lat': -29.883333, 'lon': 31.049999}  # Durban
print(find_closest_lat_lon(site_list, point_to_find))
print(find_furthest_lat_lon(site_list, point_to_find))
print('')

point_lat = point_to_find['lat']
print(point_lat)
point_lon = point_to_find['lon']
print(point_lon)
print('')

nearest_coord = find_closest_lat_lon(site_list, point_to_find)
print(nearest_coord)
nearest_lat = nearest_coord['lat']
print(nearest_lat)
nearest_lon = nearest_coord['lon']
print(nearest_lon)
print('')

#Mark nearest point

kzn_map.add_child(
    folium.features.CircleMarker(
        [nearest_lat, nearest_lon],
        radius=2,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
#        popup=label
    )
)

#Generate line to nearest point

closest_line=folium.PolyLine(locations=([point_lat, point_lon], [nearest_lat, nearest_lon]), weight=3)
kzn_map.add_child(closest_line)

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

#Use function to find distance to closest point in practice scenario

distance_site = calculate_distance(point_lat, point_lon, nearest_lat, nearest_lon)
n = distance_site
print(f'distance to nearest site is {n}km')
print('')

#Add distance marker

distance_marker = folium.Marker(
   [point_lat, point_lon],
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_site),
       )
   )

kzn_map.add_child(distance_marker)

kzn_map.save("map9.html")
webbrowser.open("map9.html")

#Code for extracting Max DL Throughput point with its coordinates

max_throughput = df['MeanUserDataRateKbps'].max()

id_max_throughput = ''

for id, rec in enumerate (df['MeanUserDataRateKbps']):
    if rec == max_throughput:
        print(id)
        id_max_throughput = id

print('')

print(id_max_throughput)

print('')

identified_max_throughput_result = df.loc[id_max_throughput].at["MeanUserDataRateKbps"]
print(identified_max_throughput_result)
identified_max_throughput_lat = df.loc[id_max_throughput].at["Latitude"]
print(identified_max_throughput_lat)
identified_max_throughput_lon = df.loc[id_max_throughput].at["Longitude"]
print(identified_max_throughput_lon)

print('')

#Code for extracting Min DL Throughput point with its coordinates

min_throughput = df['MeanUserDataRateKbps'].min()

id_min_throughput = ''

for id, rec in enumerate (df['MeanUserDataRateKbps']):
    if rec == min_throughput:
        print(id)
        id_min_throughput = id

print('')

print(id_min_throughput)

print('')

identified_min_throughput_result = df.loc[id_min_throughput].at["MeanUserDataRateKbps"]
print(identified_min_throughput_result)
identified_min_throughput_lat = df.loc[id_min_throughput].at["Latitude"]
print(identified_min_throughput_lat)
identified_min_throughput_lon = df.loc[id_min_throughput].at["Longitude"]
print(identified_min_throughput_lon)

print('')

#Plot all KZN sites with custom icons on a map, centred on Max DL Throughput point

durban_map10 = folium.Map(location = [identified_max_throughput_lat, identified_max_throughput_lon], zoom_start=18, control_scale=True)

kzn_sites4 = folium.map.FeatureGroup()

for lat, long, label in zip(sites_df.Latitude, sites_df.Longitude, sites_df.ENODEB_NAME):
    kzn_sites4.add_child(
        folium.Marker(
            [lat, long],
            popup=label,
            icon=folium.features.CustomIcon(icon_image=icon_path ,icon_size=(50,50))
        )
    )

durban_map10.add_child(kzn_sites4)

#Plot Max DL Throughput point on map

point_coordinate = [identified_max_throughput_lat, identified_max_throughput_lon]

id_circle = folium.Circle(point_coordinate, radius=100, color='#d35400', fill=True).add_child(folium.Popup('Max DL Throughput: {s}kbps'.format(s=identified_max_throughput_result)))
durban_map10.add_child(id_circle)

#durban_map10.save("map10.html")
#webbrowser.open("map10.html")

#For Max DL Troughput point find closest coordinates of three test sites

RAN_Marine_Dr_North_KZN = {'lat': -29.9006472222222, 'lon': 31.0381}
RAN_Ansteys_Beach_KZN = {'lat': -29.912917, 'lon': 31.029139}
RAN_Brighton_Beach_East_KZN = {'lat': -29.929976, 'lon': 31.010907}

site_list = [RAN_Marine_Dr_North_KZN, RAN_Ansteys_Beach_KZN, RAN_Brighton_Beach_East_KZN]

print(site_list)

print('')

max_dl_throughput_example_point_to_find = {'lat': identified_max_throughput_lat, 'lon': identified_max_throughput_lon}
print(find_closest_lat_lon(site_list, max_dl_throughput_example_point_to_find))

print('')

#Represent coordinates for all sites, then find closest one to max DL throughput point

sites_check = {}

t = 0

while t < len(df['DatasourceId']):
    sites_check[t] = {'lat': df.loc[t].at["Latitude"], 'lon': df.loc[t].at["Longitude"]}
#    print(sites_check[t])
    t = t+1

print('')

#print(sites_check)

print('')

max_dl_throughput_point_to_find = {'lat': identified_max_throughput_lat, 'lon': identified_max_throughput_lon}
#print(find_closest_lat_lon(sites_check, max_dl_throughput_point_to_find))

print('')

#Find closest coordinates to min DL throughput point

max_dl_throughput_point_to_find = {'lat': identified_max_throughput_lat, 'lon': identified_max_throughput_lon}

#Check for extracting lat and long values from dictionary

lats = []
longs = []

lats = [100, 101]
longs = [102, 103]

sites_check2 = {}
sites_check2[0] = {'lat': -29.9006472222222, 'lon': 31.0381}
sites_check2[1] = {'lat': -29.912917, 'lon': 31.029139}
print(sites_check2)

print('')

#for k in sites_check2[0]:
#    print(k)

#for k in sites_check2[0].keys():
#    print(k)
#    lats[0] = k
#print(list(k2.keys()))

for v in sites_check2[0].values():
    print(v)
    longs[0] = v

print('')
#print(lats)
#print(longs)

print('')

#for keys, values in sites_check2[0]:
#    print(keys)
#    print(values)

#print(find_closest_lat_lon(sites_check2, max_dl_throughput_point_to_find))

#Iterating to get exact lats and longs

sites_check3_lat = {}

for id, rec in enumerate (df['Latitude']):
#    sites_check3[id] = {'lat': record}
    sites_check3_lat[id] = rec

###print(sites_check3_lat)

print('')

sites_check3_lon = {}

for id, rec in enumerate (df['Longitude']):
#    sites_check3[id] = {'lat': record}
    sites_check3_lon[id] = rec

###print(sites_check3_lon)

print('')

sites_check3 = {}

#for i in range(1, n + 1):
#    result += i

lats_check = []
longs_check = []

u = 0
v = len(df['Latitude'])

lats_check = [None] * v
longs_check = [None] * v

###print(v)

print('')

result = 0

for u in range(0, v):
    lats_check[u] = sites_check3_lat[u]
    longs_check[u] = sites_check3_lon[u]
#    result += 1

#while u < len(df['Latitude']):
#    lats_check[u] = ''
#    u = u+1

###print(lats_check)

print('')

###print(longs_check)

print('')

print(lats_check[0])
print(longs_check[0])

print('')

print(lats_check[v-1])
print(longs_check[v-1])

print('')

#print(result)

#Extracting sites lats and longs into a dictionary

sites_checked = {}
#sites_checked = []

#sites_checked = {None} * v

w = 0

for w in range(0, v):
    sites_checked[w] = {'lat': lats_check[w], 'lon': longs_check[w]}

###print(sites_checked)

# Using key and value extraction to get the site coordinates in a list

x = ''

coordinates = []

y = len(df['Latitude'])

coordinates = [None] * y

#for k in sites_checked.keys():
#    print(k)
#    latitudes[k] = k

#print(latitudes)

z = 0

for v in sites_checked.values():
#    print(v)
    coordinates[z] = v
    z+=1
#    longitudes[v] = v

print('')

##print(coordinates)

print('')

#print(longitudes)

#Use function to find closest site coordinate to max DL throughput point

#nearest_site_coord = find_closest_lat_lon(coordinates, max_dl_throughput_point_to_find)
nearest_site_coord = find_closest_lat_lon(site_list, max_dl_throughput_point_to_find)
#nearest_site_coord = find_furthest_lat_lon(site_list, max_dl_throughput_point_to_find)
#nearest_site_coord = find_furthest_lat_lon(coordinates, max_dl_throughput_point_to_find)
print(nearest_site_coord)
print('')

nearest_site_lat = nearest_site_coord['lat']
nearest_site_lon = nearest_site_coord['lon']

#Mark nearest site point

##durban_map10.add_child(
##    folium.features.CircleMarker(
##       [nearest_site_lat, nearest_site_lon],
##        radius=2,
##        color='red',
##        fill=True,
##        fill_color='red',
##        fill_opacity=0.6,
#        popup=label
##    )
##)

#Generate line to nearest site point

closest_site_line=folium.PolyLine(locations=([identified_max_throughput_lat, identified_max_throughput_lon], [nearest_site_lat, nearest_site_lon]), weight=3)
durban_map10.add_child(closest_site_line)

#Use function to find distance to closest site point

distance_to_site = calculate_distance(identified_max_throughput_lat, identified_max_throughput_lon, nearest_site_lat, nearest_site_lon)
x = distance_to_site
print(f'distance to nearest site is {x}km')
#print('')

#Add distance marker for the line

site_distance_marker = folium.Marker(
   [identified_max_throughput_lat, identified_max_throughput_lon],
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "Closest site is {:10.2f} km away".format(distance_to_site),
       )
   )

durban_map10.add_child(site_distance_marker)

#Add title to map

map_title = "KZN Max DL Throughput Point"
#title_html = f'<h1 style="position:absolute;z-index:100000;left:40vw" >{map_title}</h1>'
title_html = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{map_title}</h1>'
durban_map10.get_root().html.add_child(folium.Element(title_html))

#Add text overlay to map

W, H = (300,200)
im = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg = "CLICK ON CIRCLE POINT AND SITE ICONS FOR DETAILS!"
w, h = draw.textsize(msg)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im.crop((0, 0,2*w,2*h)).save("pycoatextlogo.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo.png", bottom=10, left=0).add_to(durban_map10)

W, H = (300,200)
im6 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im6)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg6 = "ZOOM OUT TO SEE MORE SITES AROUND THE POINT!"
w, h = draw.textsize(msg6)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg6,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im6.crop((0, 0,2*w,2*h)).save("pycoatextlogo6.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo6.png", bottom=7.5, left=0).add_to(durban_map10)

durban_map10.save("map10.html")
webbrowser.open("map10.html")

#Use function to find closest site coordinate to min DL throughput point

identified_min_throughput_result = df.loc[id_min_throughput].at["MeanUserDataRateKbps"]
print(identified_min_throughput_result)
identified_min_throughput_lat = df.loc[id_min_throughput].at["Latitude"]
print(identified_min_throughput_lat)
identified_min_throughput_lon = df.loc[id_min_throughput].at["Longitude"]
print(identified_min_throughput_lon)

min_dl_throughput_point_to_find = {'lat': identified_min_throughput_lat, 'lon': identified_min_throughput_lon}

#min_nearest_site_coord = find_closest_lat_lon(coordinates, min_dl_throughput_point_to_find)
min_nearest_site_coord = find_closest_lat_lon(site_list, min_dl_throughput_point_to_find)
print(min_nearest_site_coord)
print('')

min_nearest_site_lat = min_nearest_site_coord['lat']
min_nearest_site_lon = min_nearest_site_coord['lon']

#Plot all KZN sites with custom icons on a map, centred on Min DL Throughput point

durban_map11 = folium.Map(location = [identified_min_throughput_lat, identified_min_throughput_lon], zoom_start=17, control_scale=True)

kzn_sites5 = folium.map.FeatureGroup()

for lat, long, label in zip(sites_df.Latitude, sites_df.Longitude, sites_df.ENODEB_NAME):
    kzn_sites5.add_child(
        folium.Marker(
            [lat, long],
            popup=label,
            icon=folium.features.CustomIcon(icon_image=icon_path ,icon_size=(50,50))
        )
    )

durban_map11.add_child(kzn_sites5)

#Plot Min DL Throughput point on map

min_point_coordinate = [identified_min_throughput_lat, identified_min_throughput_lon]

min_id_circle = folium.Circle(min_point_coordinate, radius=100, color='#d35400', fill=True).add_child(folium.Popup('Min DL Throughput: {s}kbps'.format(s=identified_min_throughput_result)))
durban_map11.add_child(min_id_circle)

#durban_map11.save("map11.html")
#webbrowser.open("map11.html")

#Use function to find closest site coordinate to min DL throughput point

#min_nearest_site_coord = find_closest_lat_lon(coordinates, min_dl_throughput_point_to_find)
min_nearest_site_coord = find_closest_lat_lon(site_list, min_dl_throughput_point_to_find)
print(min_nearest_site_coord)
print('')

min_nearest_site_lat = min_nearest_site_coord['lat']
min_nearest_site_lon = min_nearest_site_coord['lon']

#Generate line to nearest site point

min_closest_site_line=folium.PolyLine(locations=([identified_min_throughput_lat, identified_min_throughput_lon], [min_nearest_site_lat, min_nearest_site_lon]), weight=3)
durban_map11.add_child(min_closest_site_line)

#Use function to find distance to closest site point

min_distance_to_site = calculate_distance(identified_min_throughput_lat, identified_min_throughput_lon, min_nearest_site_lat, min_nearest_site_lon)
xx = min_distance_to_site
print(f'distance to nearest site is {xx}km')
#print('')

#Add distance marker for the line

min_site_distance_marker = folium.Marker(
   [identified_min_throughput_lat, identified_min_throughput_lon],
   icon=DivIcon(
       icon_size=(20,20),
       icon_anchor=(0,0),
       html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "Closest site is {:10.2f} km away".format(min_distance_to_site),
       )
   )

durban_map11.add_child(min_site_distance_marker)

#durban_map11.save("map11.html")
#webbrowser.open("map11.html")

#Add title to map

min_map_title = "KZN Min DL Throughput Point"
#title_html = f'<h1 style="position:absolute;z-index:100000;left:40vw" >{map_title}</h1>'
min_title_html = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{min_map_title}</h1>'
durban_map11.get_root().html.add_child(folium.Element(min_title_html))

#Add text overlay to map

W, H = (300,200)
im7 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im7)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg7 = "CLICK ON CIRCLE POINT AND SITE ICONS FOR DETAILS!"
w, h = draw.textsize(msg7)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg7,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im7.crop((0, 0,2*w,2*h)).save("pycoatextlogo7.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo7.png", bottom=10, left=0).add_to(durban_map11)

W, H = (300,200)
im8 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im8)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg8 = "ZOOM OUT TO SEE MORE SITES AROUND THE POINT!"
w, h = draw.textsize(msg8)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg8,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im8.crop((0, 0,2*w,2*h)).save("pycoatextlogo8.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo8.png", bottom=7.5, left=0).add_to(durban_map11)

durban_map11.save("map11.html")
webbrowser.open("map11.html")

print('')
print(site_list)
aa = type(site_list)
print(aa)

print('')
print(coordinates)
ab = type(coordinates)
print(ab)

print('')

#check if distance functions work in example

ac = 3

coordinates_check = []*ac
coordinates_check = [coordinates[0], coordinates[1], coordinates[2]]
ad = type(coordinates_check)
print(coordinates_check)
print(ad)

nearest_site_coord_checked = find_closest_lat_lon(coordinates_check, max_dl_throughput_point_to_find)
print(nearest_site_coord_checked)
print('')

nearest_site_checked_lat = nearest_site_coord_checked['lat']
nearest_site_checked_lon = nearest_site_coord_checked['lon']

distance_to_site_checked = calculate_distance(identified_max_throughput_lat, identified_max_throughput_lon, nearest_site_checked_lat, nearest_site_checked_lon)
print(distance_to_site_checked)

print('')

#check if distance functions work in this scenario

ae = len(coordinates)
print(ae)
print('')

#coordinates_check2 = []*ae
#coordinates_check2 = []
#coordinates_check2 = {}*ae
coordinates_check2 = {}

print(coordinates_check2)

w = 0

for w in range(0, ae):
    coordinates_check2[w] = coordinates[w]
##    print(coordinates_check2[w])
##    print(coordinates[w])

#convert dictionary to list in order for functions to work

coordinates_check3 = []

coordinates_check3 = [None] * y

ag = 0

for v in coordinates_check2.values():
    print(v)
    coordinates_check3[ag] = v
    ag+=1

ah = type(coordinates_check3)
print(coordinates_check3)
print(ah)

print(len(coordinates_check3))

##for length in range(0, ae-1):
#    coordinates_check2[length] = coordinates[length]
##    coordinates_check2 = coordinates[length]

##ag = 0

##for v in coordinates.values():
#    print(v)
##    coordinates_check2[ag] = v
##    ag+=1
#    longitudes[v] = v

af = type(coordinates_check2)
#print(coordinates_check2)
#print(af)

nearest_site_coord_checked2 = find_closest_lat_lon(coordinates_check2, max_dl_throughput_point_to_find)
print(nearest_site_coord_checked2)
print('')

##nearest_site_checked_lat2 = nearest_site_coord_checked2['lat']
##nearest_site_checked_lon2 = nearest_site_coord_checked2['lon']

##distance_to_site_checked2 = calculate_distance(identified_max_throughput_lat, identified_max_throughput_lon, nearest_site_checked_lat2, nearest_site_checked_lon2)
##print(distance_to_site_checked2)

#trying functions again with latest list generated

nearest_site_coord_checked3 = find_closest_lat_lon(coordinates_check3, max_dl_throughput_point_to_find)
print(nearest_site_coord_checked3)
print('')

nearest_site_checked_lat3 = nearest_site_coord_checked3['lat']
nearest_site_checked_lon3 = nearest_site_coord_checked3['lon']

distance_to_site_checked3 = calculate_distance(identified_max_throughput_lat, identified_max_throughput_lon, nearest_site_checked_lat3, nearest_site_checked_lon3)
print(distance_to_site_checked3)
print('')

#Further practice generating nearest and furthest point

RAN_Marine_Dr_North_KZN2 = {'lat': -29.9006472222222, 'lon': 31.0381}
RAN_Ansteys_Beach_KZN2 = {'lat': -29.912917, 'lon': 31.029139}
RAN_Brighton_Beach_East_KZN2 = {'lat': -29.929976, 'lon': 31.010907}

site_list2 = [RAN_Marine_Dr_North_KZN2, RAN_Ansteys_Beach_KZN2, RAN_Brighton_Beach_East_KZN2]

point_to_find_durban = {'lat': -29.883333, 'lon': 31.049999}  # Durban
print(find_closest_lat_lon(site_list2, point_to_find_durban))
print(find_furthest_lat_lon(site_list2, point_to_find_durban))

#Iterating once more to try get exact lats and longs

sites_checks_lat = {}

for id, rec in enumerate (sites_df['ENODEB_NAME']):
#    sites_check3[id] = {'lat': record}
    sites_checks_lat[id] = rec

###print(sites_checks_lat)

print('')

sites_checks_lon = {}

for id, rec in enumerate (sites_df['ENODEB_NAME']):
#    sites_check3[id] = {'lat': record}
    sites_checks_lon[id] = rec

v = len(sites_df['ENODEB_NAME'])

sites_checker = {}

#sites_checker = {None} * v

lats_checker = [None] * v
longs_checker = [None] * v

for u in range(0, v):
    lats_checker[u] = sites_checks_lat[u]
    longs_checker[u] = sites_checks_lon[u]

w = 0

for w in range(0, v):
    sites_checker[w] = {'lat': lats_checker[w], 'lon': longs_checker[w]}

coords = []

coords = [None] * v

z = 0

for v in sites_checker.values():
#    print(v)
    coords[z] = v
    z+=1

print(coords)

##for length in range(0, len(sites_df['ENODEB_NAME'])):
#    coordinates_check2[length] = coordinates[length]
##    coordinates_check2 = coordinates[length]

#Finding closest site to Max DL Throughput point

#Plot all KZN sites with custom icons on a map, centred on Max DL Throughput point

durban_map12 = folium.Map(location = [identified_max_throughput_lat, identified_max_throughput_lon], zoom_start=18, control_scale=True)

kzn_sites6 = folium.map.FeatureGroup()

for lat, long, label in zip(sites_df.Latitude, sites_df.Longitude, sites_df.ENODEB_NAME):
    kzn_sites6.add_child(
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
#    print(sites_check[t])
    t = t+1

#print('')

#print(sites_check)

site_coordinates_check4 = []

site_coordinates_check4 = [None] * len(sites_df['ENODEB_NAME'])

ao = 0

for v in sites_check.values():
#    print(v)
    site_coordinates_check4[ao] = v
    ao+=1

#print('')

print(site_coordinates_check4)
ap = type(site_coordinates_check4)
print(ap)

print('')

max_dl_throughput_point_to_be_found = {'lat': identified_max_throughput_lat, 'lon': identified_max_throughput_lon}
nearest_site_coord_checked4 = find_closest_lat_lon(site_coordinates_check4, max_dl_throughput_point_to_be_found)
print(nearest_site_coord_checked4)

print('')

nearest_site_checked_lat4 = nearest_site_coord_checked4['lat']
nearest_site_checked_lon4 = nearest_site_coord_checked4['lon']

#print(distance_to_site_checked4)
#print('')

#Generate line to nearest site point

max_closest_site_line=folium.PolyLine(locations=([identified_max_throughput_lat, identified_max_throughput_lon], [nearest_site_checked_lat4, nearest_site_checked_lon4]), weight=3)
durban_map12.add_child(max_closest_site_line)

#Use function to find distance to closest site point

distance_to_site_checked4 = calculate_distance(identified_max_throughput_lat, identified_max_throughput_lon, nearest_site_checked_lat4, nearest_site_checked_lon4)
#x = distance_to_site
print(f'distance to nearest site is {distance_to_site_checked4}km')
print('')

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
#title_html = f'<h1 style="position:absolute;z-index:100000;left:40vw" >{map_title}</h1>'
max_title_html = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{max_map_title}</h1>'
durban_map12.get_root().html.add_child(folium.Element(max_title_html))

#Add text overlay to map

W, H = (300,200)
im9 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im9)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg9 = "CLICK ON CIRCLE POINT AND SITE ICONS FOR DETAILS!"
w, h = draw.textsize(msg9)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg9,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im9.crop((0, 0,2*w,2*h)).save("pycoatextlogo9.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo9.png", bottom=10, left=0).add_to(durban_map12)

W, H = (300,200)
im10 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im10)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg10 = "ZOOM OUT TO SEE MORE SITES AROUND THE POINT!"
w, h = draw.textsize(msg10)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg10,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im10.crop((0, 0,2*w,2*h)).save("pycoatextlogo10.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo10.png", bottom=7.5, left=0).add_to(durban_map12)

durban_map12.save("map12.html")
webbrowser.open("map12.html")

#Finding closest site to Min DL Throughput point

#Plot all KZN sites with custom icons on a map, centred on Min DL Throughput point

durban_map13 = folium.Map(location = [identified_min_throughput_lat, identified_min_throughput_lon], zoom_start=17, control_scale=True)

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
#    print(sites_check[t])
    t2 = t2+1

#print('')

#print(sites_check)

site_coordinates_check5 = []

site_coordinates_check5 = [None] * len(sites_df['ENODEB_NAME'])

ao2 = 0

for v in sites_check.values():
#    print(v)
    site_coordinates_check5[ao2] = v
    ao2+=1

#print('')

print(site_coordinates_check5)
ap2 = type(site_coordinates_check5)
print(ap2)

print('')

min_dl_throughput_point_to_be_found = {'lat': identified_min_throughput_lat, 'lon': identified_min_throughput_lon}
nearest_site_coord_checked5 = find_closest_lat_lon(site_coordinates_check5, min_dl_throughput_point_to_be_found)
print(nearest_site_coord_checked5)

print('')

nearest_site_checked_lat5 = nearest_site_coord_checked5['lat']
nearest_site_checked_lon5 = nearest_site_coord_checked5['lon']

#Generate line to nearest site point

min_closest_site_line=folium.PolyLine(locations=([identified_min_throughput_lat, identified_min_throughput_lon], [nearest_site_checked_lat5, nearest_site_checked_lon5]), weight=3)
durban_map13.add_child(min_closest_site_line)

#Use function to find distance to closest site point

distance_to_site_checked5 = calculate_distance(identified_min_throughput_lat, identified_min_throughput_lon, nearest_site_checked_lat5, nearest_site_checked_lon5)
#x = distance_to_site
print(f'distance to nearest site is {distance_to_site_checked5}km')
print('')

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
#title_html = f'<h1 style="position:absolute;z-index:100000;left:40vw" >{map_title}</h1>'
min_title_html = f'<h1 style="position:absolute;z-index:100000;left:30vw" >{min_map_title}</h1>'
durban_map13.get_root().html.add_child(folium.Element(min_title_html))

#Add text overlay to map

W, H = (300,200)
im11 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im11)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg11 = "CLICK ON CIRCLE POINT AND SITE ICONS FOR DETAILS!"
w, h = draw.textsize(msg11)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg11,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im11.crop((0, 0,2*w,2*h)).save("pycoatextlogo11.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo11.png", bottom=10, left=0).add_to(durban_map13)

W, H = (300,200)
im12 = Image.new("RGBA",(W,H))
draw = ImageDraw.Draw(im12)
#msg = "pycoa.fr (data from: {})".format(mypandas.data_base)
msg12 = "ZOOM OUT TO SEE MORE SITES AROUND THE POINT!"
w, h = draw.textsize(msg12)
#fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 14)
#draw.text((0,0), msg, font=fnt,fill=(0, 0, 0))
draw.text((0,0), msg12,fill=(0, 0, 0))
#draw.text((0,30), msg,fill=(0, 0, 0))
im12.crop((0, 0,2*w,2*h)).save("pycoatextlogo12.png", "PNG")
#FloatImage("pycoatextlogo.png", bottom=0, left=0).add_to(durban_map10)
FloatImage("pycoatextlogo12.png", bottom=7.5, left=0).add_to(durban_map13)

durban_map13.save("map13.html")
webbrowser.open("map13.html")

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
                                # If a specific region was selected, show the Success vs. Failed counts for the region
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Throughput range (kbps):"),
                                # TASK 3: Add a slider to select throughput
                                #dcc.RangeSlider(id='throughput-slider',...)
                                #html.Div(dcc.RangeSlider(id='throughput-slider',
                                dcc.RangeSlider(id='throughput-slider',
	                                min=0, max=600000, step=50000,
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

                                #Add a boxplot for throughput per area
#                                html.Div(dcc.Graph(id='throughput-box-plot')),
#                                html.Br(),

                                #Add a violin plot for throughput per area
                                html.Div(dcc.Graph(id='throughput-violin-plot')),
                                html.Br(),

                                #Add saved Folium map plot of KZN DL Failure Test Points
#                               html.P("Map Plot", style={"font-weight": "bold", "fontSize": 20}),
                                html.P("Map 1 -> Zoom in and click points to view their logfile name details", style={"fontSize": 20}),
#                                html.Iframe(srcDoc = open('map2.html', 'r').read(), style={'width': '800px', 'height': '500px'}),
                                html.Iframe(srcDoc = open('map2.html', 'r').read(), style={'width': '1050px', 'height': '510px'}),
                                html.Br(),

                                # Add saved Folium map plot of KZN DL Testing Status Markers
                                html.P("Map 2 -> Click marker clusters to focus on specific areas - green markers are successful samples and red markers are failure samples", style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map3.html', 'r').read(),
                                            style={'width': '1050px', 'height': '510px'}),
                                html.Br(),

                                # Add saved Folium map plot of KZN DL Throughput Heatmap
                                html.P("Map 3 -> Zoom in for better granularity view of areas, with darker shading indicating higher throughput areas", style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map5.html', 'r').read(),
                                            style={'width': '1050px', 'height': '510px'}),
                                html.Br(),

                                # Add saved Folium map plot of KZN DL Radio Access Technologies
                                html.P("Map 4 -> Points indicate serving technology of the sample - use legend to determine the tech", style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map6.html', 'r').read(),
                                            style={'width': '1050px', 'height': '510px'}),
                                html.Br(),

                                # Add saved Folium map plot of KZN Max DL Throughput Point
                                html.P("Map 5 -> Click on max DL throughput point and site icons for further details", style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map12.html', 'r').read(),
                                            style={'width': '1050px', 'height': '510px'}),
                                html.Br(),

                                # Add saved Folium map plot of KZN Min DL Throughput Point
                                html.P("Map 6 -> Click on min DL throughput point and site icons for further details", style={"fontSize": 20}),
                                html.Iframe(srcDoc=open('map13.html', 'r').read(),
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
                      title='KZN Mean DL Throughput Per Area')
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

# Add a callback function for `throughput-box-plot` as output
#@app.callback(Output(component_id='throughput-box-plot', component_property='figure'),
#              Input(component_id="throughput-slider", component_property="value"))
#def get_box_plot(throughput_slider):
#    filtered_df9 = df[(df['MeanUserDataRateKbps']>=throughput_slider[0]) & (df['MeanUserDataRateKbps']<=throughput_slider[1])]
#    fig10 = px.box(filtered_df9, x='Area', y='MeanUserDataRateKbps', color='Area',
#                      title='Throughput Boxplot Per Area')
#    return fig10

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
