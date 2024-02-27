import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import os
from function.function import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pydeck as pdk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


@st.cache_resource
def random_forest_fit(X_train, X_test, y_train, y_test, n_estimators, max_depth, min_samples_split):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    rf.fit(X_train, y_train)
    return rf



try:
    col = st.columns((1, 1), gap='small') # page slice 기능 () 내부에 원하는 사이즈로 설정


    x_data = st.session_state['x_col']
    y_data = st.session_state['y_col']

    st.header('Modeling')

    with col[0]:
        st.write('### x data')
        st.dataframe(x_data, width = 1000)
        st.write('---')
    with col[1]:
        st.write('### y data')
        st.write(y_data)
        st.write('---')


    st.write('---')

    st.write('### RF Classifier')

    test_size = float(st.text_input('test size set', value=0.2)) # text_input을 사용하여 입력 받기 // value는 기본값 설정

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = test_size, random_state = 42)

    n_estimators = int(st.text_input('n_estimators', value=100))
    max_depth = st.select_slider('max_depth', options=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    min_samples_split = int(st.slider('min_samples_split', min_value=2, max_value=10, value=2))


    if st.checkbox('RF Classifier fit'):
        rf = random_forest_fit(X_train, X_test, y_train, y_test, n_estimators, max_depth, min_samples_split)

    if st.checkbox('RF Classifier predict'):
        y_pred = rf.predict(X_test)
        st.write(y_pred)

    st.write('---')

    st.write('Result of RF Classifier')

    if st.checkbox('RF Classifier score'):
        score = rf.score(X_test, y_test)
        st.write(score)

    if st.checkbox('RF Classifier accuracy'):
        accuracy = accuracy_score(y_test, y_pred)
        st.write(accuracy)

    if st.checkbox('RF Classifier precision'):
        precision = precision_score(y_test, y_pred, average='weighted')
        st.write(precision)


    if st.checkbox('RF Classifier confusion_matrix'):
        confusion = confusion_matrix(y_test, y_pred)
        st.write(confusion)


    st.write('---')
    st.write('### Result visualization with PCA')

    try:
        scaler = StandardScaler()
        X = x_data.dropna()

        x_train = scaler.fit_transform(X)
        x_test = scaler.transform(X_test)

        pca = PCA(n_components=2)
        pc_x_train = pca.fit_transform(x_test)
        pc_x_test = pca.transform(x_test)

        pc_x_test = pd.DataFrame(pc_x_test, columns=['PC1', 'PC2'])
        pc_x_test['target'] = y_test
        pc_x_test['predict'] = y_pred

        scatter = alt.Chart(pc_x_test).mark_circle().encode(
            x='PC1',
            y='PC2',
            color='target'
        ).properties(
            width=1600,
            height=800
        )

        st.write(scatter)
    except:
        st.write('PCA를 위한 데이터가 없습니다.')
        st.write('예측을 진행해주세요.')

except:
    st.write('데이터를 불러오지 못했습니다.')
    st.write('EDA 페이지에서 데이터를 불러온 후 다시 시도해주세요.')