import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import os
from function.function import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pydeck as pdk



df = pd.read_csv('./data/supermarket_sales.csv')

lat_lon_data = {
    'Yangon': (16.8409, 96.1735),
    'Naypyitaw': (19.7633, 96.0785),
    'Mandalay': (21.9588, 96.0891)
}

df['Latitude'] = df['City'].map(lambda city: lat_lon_data[city][0])
df['Longitude'] = df['City'].map(lambda city: lat_lon_data[city][1])



st.set_page_config(
    page_title="Supermarket Sales Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

st.title('🛒 Supermarket Sales Dashboard')
st.subheader('Data Exploration')
st.write('---')

st.write('### Data head')
st.write(df.head())
st.write('---')

# 데이터 타입 확인
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

st.write('### Data types')
st.write('#### Numeric columns')
st.dataframe(pd.DataFrame(numeric_columns).T, width = 1000)
st.write('#### Categorical columns')
st.dataframe(pd.DataFrame(categorical_columns).T, width = 1000)


st.write('---')

st.write('### Data Visualization')

agg_method = st.selectbox('Select aggregation method', ['Average', 'Count']) # 리스트를 내부에 직접 설정
selected_col = st.selectbox('Select a column for visualization', numeric_columns) # 리스트를 변수로 받아서 설정

if agg_method == 'Average':
    city_agg = df.groupby('City')[selected_col].mean().reset_index()
    city_agg['Label'] = city_agg['City'] + ': 평균 ' + round(city_agg[selected_col], 2).astype(str)
elif agg_method == 'Count':
    city_agg = df['City'].value_counts().reset_index()
    city_agg.columns = ['City', 'Count']
    city_agg['Label'] = city_agg['City'] + ': 카운트 ' + city_agg['Count'].astype(str)

# 위도 경도 정보 추가
city_agg = city_agg.merge(pd.DataFrame.from_dict(lat_lon_data, orient='index', columns=['Latitude', 'Longitude']).reset_index(), how='left', left_on='City', right_on='index')

scale_factor = 500  # 반경 계산을 위한 스케일 팩터
elevation_factor = 1000  # 높이 계산을 위한 스케일 팩터

if agg_method == 'Average':
    city_agg['radius'] = city_agg[selected_col] * scale_factor
    city_agg['elevation'] = city_agg[selected_col] * elevation_factor
elif agg_method == 'Count':
    city_agg['radius'] = city_agg['Count'] * scale_factor
    city_agg['elevation'] = city_agg['Count'] * elevation_factor


# Pydeck 차트
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=19.7633, longitude=96.0785, zoom=5, pitch=70), # 초기 위,경도, 확대 정도, 각도
    layers=[
        pdk.Layer(
            'HexagonLayer', # 모양 그래프 설정
            data=city_agg,
            get_position='[Longitude, Latitude]',
            get_elevation='elevation',  # 동적 높이 설정
            elevation_scale=50000,  # 높이 스케일 조절
            elevation_range=[0, 300000],  # 최소 및 최대 높이 범위
            radius=20000,  # 헥사곤 반경
            extruded=True,  # 3D 헥사곤 사용 여부
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer", # text 사용 여부
            data=city_agg,
            get_position='[Longitude, Latitude]',
            get_text='Label',
            get_size=30,
            get_color=[0, 0, 0, 200],  # 텍스트 색상: 검정
            get_angle=0,
            get_alignment_baseline="'bottom'",
        )
    ],
    width=1000,
    height=1000,
))

# st.write(city_agg['elevation'])

st.write('---')

