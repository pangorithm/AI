#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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

# 데이터 정규화
x = (x - x.mean())/(x.std())
print(x.describe())

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
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=11, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

# 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

##earlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor= 'val_loss', patience=100, mode='auto',
                              verbose=1, restore_best_weights=True ) # restore_best_weights의 기본값은 false이므로 true로 반드시 변경

# Model Check point
# Model Check point
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./mcp/noshow02.hdf5'
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[earlyStopping, mcp],verbose=1)
end_time = time.time() - start_time

model.summary()
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('소요시간 : ', end_time)
'''
8842/8842 [==============================] - 35s 4ms/step - loss: 0.4481 - accuracy: 0.7996 - val_loss: 0.4476 - val_accuracy: 0.7998  
Epoch 00259: early stopping
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 16)                192
_________________________________________________________________
dense_1 (Dense)              (None, 32)                544
_________________________________________________________________
dropout (Dropout)            (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 64)                2112
_________________________________________________________________
dense_3 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 33
=================================================================
Total params: 4,961
Trainable params: 4,961
Non-trainable params: 0
_________________________________________________________________
691/691 [==============================] - 2s 3ms/step - loss: 0.4549 - accuracy: 0.7942
loss :  0.4549362361431122
acc :  0.794173538684845
소요시간 :  8936.668451786041
'''