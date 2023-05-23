import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


##### 데이터 전처리 시작 #####

path = './_data/'
datasets = pd.read_csv(path + 'medical_noshow.csv')

print('columns : \n',datasets.columns)
print('head : \n',datasets.head(7))

x = datasets[['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',
       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = datasets[['No-show']]
print(x.shape, y.shape)

print(x.info())  # info() 컬럼명, null값, data타입 확인
print(x.describe())

# 결과에 영향을 주지 않는 값 삭제
x = x.drop(['PatientId', 'AppointmentID'], axis=1)

print(x.shape)

# 문자를 숫자로 변경
from sklearn.preprocessing import LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index)    # 오브젝트 컬럼 리스트 추출
print(ob_col)
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)
    
print('columns : \n',x.columns)
print('head : \n',x.head(7))

###상관계수 히트맵###
import matplotlib.pyplot as plt
import seaborn as sns

# pip install seaborn
sns.set(font_scale = 1.2)
sns.set(rc = {'figure.figsize':(9, 6)})   # 히트맵 데이터 맵 말고 히트맵 그림의 크기
sns.heatmap(data = x.corr(), #상관관계
            square = True,
            annot = True,
            cbar = True,
           )

##### 전처리 완료 #####


##### 훈련 구성 시작 #####
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size = 0.2, shuffle=True, random_state=77
)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)