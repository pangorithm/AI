# [실습]
# 1. boost 계열 모델 3대장 적용 (iris, cancer, wine, california)
# 2. feature importances 확인 및 feature 정리 후 성능 비교
# 3. 팀 프로젝트에 1번, 2번 적용

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
import time

import tensorflow as tf

random_state = 42
tf.random.set_seed(random_state)


##### 데이터 전처리 시작 #####
path = '.C조teamProject/_data/'
datasets = pd.read_csv('D:/Ai_study/C조teamProject/_data/medical_noshow.csv')

# print('columns : \n',datasets.columns)
# print('head : \n',datasets.head(7))

x = datasets[['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = datasets[['No-show']]
# print(x.shape, y.shape)

# print(x.info())  # info() 컬럼명, null값, data타입 확인

# print(x.describe())

# 결과에 영향을 주지 않는 값 삭제
x = x.drop(['PatientId', 'AppointmentID'], axis=1)

# print(x.shape)

# 문자를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index)    # 오브젝트 컬럼 리스트 추출
# print(ob_col)
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
y = LabelEncoder().fit_transform(y.values)
# no = 0 , yes = 1

x = x.fillna(np.NaN)


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.2, shuffle=True, random_state=random_state
)

# kfold : scikit-learn 라이브러리에서 제공하는 교차 검증(Cross-validation) 기법
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
# KFold : 회귀문제에서 교차검증시 사용
# StratifedKFold : 레이블의 데이터가 왜곡되었거나, 일반적으로 분류에서의 교차검증에 사용

# scler 적용(스케일링)
scaler = MinMaxScaler()     # 최대 최솟값을 기준으로 데이터를 0과 1사이의 값으로 정규화한다.
# 스케일링의 종류로는 MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 등이 있다.
# scaler.fit = 데이터의 범위를 조정하기 위한 변환 계수를 계산하는 메서드
# scaler.transform 미리 학습된 변환 계수를 적용하여 데이터를 변환
x_train = scaler.fit_transform(x_train)   # scaler.fit을 수행하고 scaler.transform한 것과 같다
x_test = scaler.transform(x_test)
# 훈련 데이터에만 fit을 적용하고 테스트 데이터에 fit을 안하는 이유
# 훈련 데이터와 테스트 데이터를 함께 사용하여 전체 데이터에 대해 fit을 수행할 경우,
# 테스트 데이터가 훈련 데이터에 미리 노출되어 정보 누설이 발생할 수 있다.

### 전처리 완료 ###



# 2. 모델 구성
# from catboost import CatBoostClassifier
# model = CatBoostClassifier()
# from lightgbm import LGBMClassifier
# model = LGBMClassifier()
from xgboost import XGBClassifier
model = XGBClassifier()     
# MLP 또는 CNN에서 레이어를 직접 구성하였던것과 다르게 머신러닝 코딩에서는 이미 구현된 알고리즘 함수를 입력한다.

# 머신러닝 모델의 종류
# SVM : (분류 모델 : SVC, 회귀 모델 : SVR, 선형 모델 : linear)
# tree : (DecisionTreeClassifier : 분류 문제, DecisionTreeRegressor : 회귀 문제)
# Ensemble : (RandomForestClassifier : 의사결정트리 기반 분류 앙상블 모델, RandomForestRegressor: 회귀 앙상블 모델)
# All_Estimator : (옵션에 따라서 해당 분류에 해당하는 sklearn의 모델 전부 테스트)

# 3. 훈련
model.fit(x_train, y_train)

##earlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor= 'val_loss', patience=100, mode='auto',
                              verbose=1, restore_best_weights=True ) # restore_best_weights의 기본값은 false이므로 true로 반드시 변경

# 4. 평가, 예측
from sklearn.model_selection import cross_val_score, cross_val_predict
score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv = cross validation
# kfold를 통해 여러개로 분할했던 훈련 및 검증 데이터들을 통해 val_acc값을 교차검증
print('cross validation acc : ' , score)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold) #모델과 테스트 데이터 및 교차검증 데이터를 통해 데이터 예상 
print('cv pred : ', y_predict)
acc = accuracy_score(y_test, y_predict)   # 테스트 데이터와 예상 데이터의 acc 값 연산
print('cv pred acc : ', acc)

# feature 정리 전
# cv pred :  [0 0 0 ... 0 0 0]
# cv pred acc :  0.7915498054826744

# 시각화
import matplotlib.pyplot as plt
n_features = x.shape[1]     # x의 열의 갯수를 나타내는 x.shape[1]을 통해 특성의 개수를 계산
plt.barh(range(n_features), model.feature_importances_, align='center')      #막대그래프 생성
# 인자 순서대로 x축의 위치설정, y값의 높이 설정, 가운데 정렬
# model.feature_importances_ : 해당 모델의 특성 중요도를 나타내는 값들을 반환한다.
# 각 특성의 중요도는 머신러닝 모델의 종류에 따라서 다르다

# plt.yticks(np.arange(n_features), datasets.feature_names)
plt.yticks(np.arange(n_features), ['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'])
# 인자 순서대로 y축 눈금의 위치와 레이블 설정, 리스트 내의 문자열들(datasets.feature_names)을 표시

plt.title('medical_noshow') # 제목 표기
plt.ylabel('feature')       # y축의 레이블 표기
plt.xlabel('importances')   # x축의 레이블 표기
plt.ylim(-1, n_features)    # y축의 범위를 -1에서 n_features까지로 설정
plt.show()                  #그래프를 화면에 표시

# feature 정리 후

# cat boost
# cv pred :  [0 0 0 ... 0 0 0]
# cv pred acc :  0.7915498054826744

# LGBM
# cross validation acc :  [0.8040147  0.80134585 0.80032798 0.79851843 0.80191133]
# cv pred :  [0 0 0 ... 0 0 0]
# cv pred acc :  0.7932687958020447

# XGB
# cv pred :  [0 0 0 ... 0 0 0]
# cv pred acc :  0.7861214150004524