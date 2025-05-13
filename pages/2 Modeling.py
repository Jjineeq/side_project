import streamlit as st
import pandas as pd
import altair as alt
from function.function import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

@st.cache_resource # 캐시 사용해서 학습한 모델 저장 // 함수로 만든 코드만 가능
def random_forest_fit(X_train, X_test, y_train, y_test, n_estimators, max_depth, min_samples_split):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split) # 분류 문제를 해결하기 위해 분류 모델 정의
    rf.fit(X_train, y_train) # 모델 학습
    return rf

try: # 데이터 불러오기
    col = st.columns((1, 1), gap='small') # page slice 기능 () 내부에 원하는 사이즈로 설정

    # EDA.py 마지막에 저장한 x_col, y_col을 session_state에서 받아옴
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

    test_size = float(st.text_input('test size set', value=0.2)) # text_input을 사용하여 입력 받기 // value는 기본값 설정이며, text으로 str형식으로 정의됨 --> 숫자가 필요하므로 float로 변환

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = test_size, random_state = 42)

    n_estimators = int(st.text_input('n_estimators', value=100)) # 모델 파라미터 설정
    max_depth = st.select_slider('max_depth', options=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # 모델 파라미터 설정 // select_slider에서 options으로 범위 설정 가능
    min_samples_split = int(st.slider('min_samples_split', min_value=2, max_value=10, value=2)) # 모델 파라미터 설정 // int 형식으로 변환 필요

    if st.checkbox('RF Classifier fit'): # check box를 사용하여 학습 진행 여부 선택 -> 페이지 속도 향상
        rf = random_forest_fit(X_train, X_test, y_train, y_test, n_estimators, max_depth, min_samples_split) # 사전에 정의한 함수 사용해서 모델 학습

        if st.checkbox('RF Classifier predict'):
            y_pred = rf.predict(X_test) # 모델 테스트
            st.write(y_pred)

            st.write('---')

            st.write('Result of RF Classifier')

            if st.checkbox('RF Classifier score'):
                score = rf.score(X_test, y_test) # 정확도 측정
                st.write(score)

            if st.checkbox('RF Classifier accuracy'):
                accuracy = accuracy_score(y_test, y_pred) # accuracy 측정
                st.write(accuracy)

            if st.checkbox('RF Classifier precision'):
                precision = precision_score(y_test, y_pred, average='weighted') # precision 측정
                st.write(precision)


            if st.checkbox('RF Classifier confusion_matrix'):
                confusion = confusion_matrix(y_test, y_pred) # confusion matrix 확인
                st.write(confusion)


    st.write('---')
    st.write('### Result visualization with PCA')

    try: # 예측을 수행하지 않았을때 예외 처리를 위한 try문
        scaler = StandardScaler()
        X = x_data.dropna()

        x_train = scaler.fit_transform(X) # 표준화 적용 및 정의
        x_test = scaler.transform(X_test) # 표준화 적용

        pca = PCA(n_components=2) # PCA 차원 정의
        pc_x_train = pca.fit_transform(x_test) # PCA 정의 및 변환
        pc_x_test = pca.transform(x_test) # PCA 변환

        pc_x_test = pd.DataFrame(pc_x_test, columns=['PC1', 'PC2']) # PCA 결과 확인을 위해 DF으로 변환
        pc_x_test['target'] = y_test
        pc_x_test['predict'] = y_pred

        scatter = alt.Chart(pc_x_test).mark_circle().encode(
            x='PC1',
            y='PC2',
            color='target' # 사전에 정의한 PC1, PC2를 x,y 축으로 색을 target으로 설정
        ).properties(
            width=1600,
            height=800
        )

        st.write(scatter)
    except: # 예측을 수행하지 않았을때 예외 처리
        st.write('PCA를 위한 데이터가 없습니다.')
        st.write('예측을 진행해주세요.')

except KeyError as e: # session_state에서 x_col, y_col을 찾지 못했을때 예외 처리
    st.error(f'필요한 키를 찾을 수 없습니다 : {e}')
    st.error("EDA 페이지에서 데이터를 확인해주세요")
except FileNotFoundError as e:
    st.error(f'파일을 찾을 수 없습니다 : {e}')
except ValueError as e:
    st.error(f'잘못된 값이 입력되었습니다 : {e}')
except Exception as e:
    st.error(f'예상치 못한 오류가 발생했습니다 : {e}')