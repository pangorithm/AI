# [실습]
# 1. boost 계열 모델 3대장 적용 (iris, cancer, wine, california)
# 2. feature importances 확인 및 feature 정리 후 성능 비교
# 3. 팀 프로젝트에 1번, 2번 적용

#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
import time

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

# print('columns : \n',x.columns)
# print('head : \n',x.head(7))
# print('y : ',y[0:8])

# ###상관계수 히트맵###
# import matplotlib.pyplot as plt
# import seaborn as sns

# # pip install seaborn
# sns.set(font_scale = 1.2)
# sns.set(rc = {'figure.figsize':(9, 6)})   # 히트맵 데이터 맵 말고 히트맵 그림의 크기
# sns.heatmap(data = x.corr(), #상관관계
#             square = True,
#             annot = True,
#             cbar = True,
#            )

##### 전처리 완료 #####



##### 훈련 구성 시작 #####
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.2, shuffle=True, random_state=77
)

# kfold
n_splits = 5
random_state = 42
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# scler 적용
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
# from catboost import CatBoostClassifier
# model = CatBoostClassifier()
# from lightgbm import LGBMClassifier
# model = LGBMClassifier()
from xgboost import XGBClassifier
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)

##earlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor= 'val_loss', patience=100, mode='auto',
                              verbose=1, restore_best_weights=True ) # restore_best_weights의 기본값은 false이므로 true로 반드시 변경

# 4. 평가, 예측
from sklearn.model_selection import cross_val_score, cross_val_predict
score = cross_val_score(model, x_train, y_train, cv=kfold)   # cv = cross validation
print('cross validation acc : ' , score)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print('cv pred : ', y_predict)
acc = accuracy_score(y_test, y_predict)
print('cv pred acc : ', acc)

# feature 정리 전
# cv pred :  [0 0 0 ... 0 0 0]
# cv pred acc :  0.7915498054826744

# 시각화
import matplotlib.pyplot as plt
n_features = x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), datasets.feature_names)
plt.yticks(np.arange(n_features), ['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'])
plt.title('medical_noshow')
plt.ylabel('feature')
plt.xlabel('importances')
plt.ylim(-1, n_features)
plt.show()

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