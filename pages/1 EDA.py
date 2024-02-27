import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from function.function import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pydeck as pdk
import seaborn as sns

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

select_col_plot = st.selectbox('Select column for visualization', numeric_columns)

st.write('### Line')

if st.checkbox('Line Plot'):
    fig, ax = plt.subplots(figsize=(16, 4))

    # 각 도시별로 데이터 필터링 후, 선 그래프 그리기
    for city in df['City'].unique():
        df_city = df[df['City'] == city]
        ax.plot(df_city.index, df_city[select_col_plot], label=city)

    ax.set_xlabel('Date')
    ax.set_ylabel(select_col_plot)
    ax.set_title(f'Line Chart of {select_col_plot} Over Time by City')
    ax.legend()

    # Streamlit으로 차트 표시
    st.pyplot(fig)


st.write('### Histogram')

if st.checkbox('Hist Plot'):
    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(select_col_plot, bin=True),
        y=select_col_plot,
        color='City'
    ).properties(
        width=1600,
        height=400
    )

    # hist = px.histogram(df, x=select_col_plot, nbins=20, title='Histogram')

    st.write(hist)

st.write('### Area')

if st.checkbox('Area Plot'):
    area_chart = px.area(df, 
                         x="Date", 
                         y=select_col_plot,  
                         color="City",  # 'City'는 범주형 데이터로 색상 구분에 사용
                         title="Area Plot by City")
    
    # Streamlit에서 플롯 표시
    st.plotly_chart(area_chart, use_container_width=True)


st.write('---')

st.write('## PCA')

# 데이터 표준화
scaler = StandardScaler()

X = df[numeric_columns].dropna() # numeric columns만 추출 후 결측치 제거
X = scaler.fit_transform(X) # 표준화

pca = PCA(n_components=2) # 2차원으로 축소
X_pca = pca.fit_transform(X) # pca 진행

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['City'] = df['City']

st.dataframe(pca_df.T, height=200)

st.write('### PCA Scatter')

if st.checkbox('PCA 시각화'):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(20, 10))
    scatter_plot = sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='City', palette='deep', s=100)
    
    scatter_plot.set_title('PCA Visualization')
    scatter_plot.set_xlabel('PC1')
    scatter_plot.set_ylabel('PC2')
    plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    st.pyplot(plt)

st.write('---')

st.markdown('### :green[Modeling을 위한 column 선택]')

x_col = st.multiselect('Select columns for modeling', numeric_columns)
y_col = st.selectbox('Select a target column', df.columns)


st.write('다음 페이지에서 모델링을 수행하기 위해 선택한 column을 저장합니다.')


if st.button('Save'):
    st.session_state['x_col'] = df[x_col]
    st.session_state['y_col'] = df[y_col]

