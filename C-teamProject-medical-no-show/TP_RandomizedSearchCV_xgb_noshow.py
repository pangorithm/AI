#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# In[37]:


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
n_splits = 8
random_state = 42
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# scler 적용
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[38]:


param = {
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7, 9],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'min_child_weight': [1, 5, 10, 20],
    'gamma': [0, 0.1, 0.5, 1],
    'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1] ,
    'colsample_bynode' : [0,0.1,0.2,0.3,0.5,0.7,1]
    }


# In[39]:


# 2. 모델
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
xgb = XGBClassifier()
model = RandomizedSearchCV(xgb, param, cv = kfold, verbose = 1, refit = True, n_jobs = -1, n_iter=2048, random_state=42)


# In[40]:


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


# In[ ]:


# Fitting 8 folds for each of 2048 candidates, totalling 16384 fits
# 최적의 파라미터 :  {'subsample': 1.0, 'reg_lambda': 0.5, 'reg_alpha': 0.1, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 1.0, 'colsample_bynode': 1, 'colsample_bylevel': 1}
# 최적의 매개변수 :  XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1.0,
#               early_stopping_rounds=None, enable_categorical=False,
#               eval_metric=None, feature_types=None, gamma=0.1, gpu_id=None,
#               grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.1, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=9, max_leaves=None,
#               min_child_weight=1, missing=nan, monotone_constraints=None,
#               n_estimators=300, n_jobs=None, num_parallel_tree=None,
#               predictor=None, random_state=None, ...)
# best_score :  0.8018456949225559
# model_score :  0.8037184474803221
# 걸린 시간 :  49266.53394436836 초

