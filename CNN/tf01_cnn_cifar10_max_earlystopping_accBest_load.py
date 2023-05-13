import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import cifar10
import time

# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)

#정규화
x_train = x_train/255.0
x_test = x_test/255.0

from keras.models import load_model
model = load_model('./gitHub/AI/CNN/mcp_data/tf_cifar10_0_7994999885559082.hdf5')
# gitHub\AI\CNN\mcp_data\tf_cifar10_0_8029000163078308.hdf5
# mcp 모델을 로드할때 필요
# 불러올때는 데이터 셋과 아래의 평가, 예측 항목만 필요


# 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
# print('걸린시간 : ', end_time)

# Epoch 00135: early stopping
# 313/313 [==============================] - 2s 6ms/step - loss: 0.6033 - accuracy: 0.7960
# loss :  0.6033015847206116
# acc :  0.7960000038146973
# 걸린시간 :  2349.84707570076