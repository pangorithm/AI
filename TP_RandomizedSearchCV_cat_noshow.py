#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')



# 1. 데이터
path = '../_data/'
datasets = pd.read_csv(path + 'medical_noshow.csv')
x = datasets[['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = datasets[['No-show']]

x = x.drop(['PatientId', 'AppointmentID'], axis=1)

x = x.fillna(np.NaN)    

# 문자를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index)    # 오브젝트 컬럼 리스트 추출
print(ob_col)
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
y = LabelEncoder().fit_transform(y.values)
# no = 0 , yes = 1

x = x.fillna(np.NaN)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# kfold
n_splits = 5
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# scler 적용
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


param = {
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'depth': [3, 5, 7, 9],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'colsample_bylevel': [0.5, 0.7, 1.0],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.5, 0.7, 1.0],
    'random_strength': [0.1, 0.5, 1, 5],
    'bagging_temperature': [0.1, 0.5, 1, 5],
    'border_count': [5, 10, 20, 50]
    }


# 2. 모델
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
lgbm = CatBoostClassifier()
model = RandomizedSearchCV(lgbm, param, cv = kfold, verbose = 1, refit = True, n_jobs = -1, n_iter=50000, random_state=42)



import time
start_time = time.time()
# 3. 훈련
model.fit(x_train, y_train)
end_time = time.time() - start_time

print('최적의 파라미터 : ', model.best_params_)
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score : ', model.best_score_)
print('model_score : ', model.score(x_test, y_test))
print('걸린 시간 : ', end_time, '초')

