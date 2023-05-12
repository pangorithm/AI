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

# 모델 구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,4),
            padding='same',
            activation='relu',
            input_shape=(32, 32, 3)))
model.add(Conv2D(64, (4,2), padding='same', activation='relu'))
model.add(MaxPooling2D(4,4))
model.add(Dropout(0.35))
model.add(Conv2D(64, (3,2), padding='same', activation='relu'))
model.add(Conv2D(64, (2,3), padding='same', activation='relu'))
model.add(MaxPooling2D(3,3))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min',
                              verbose=1, restore_best_weights=True )

# model check point
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./mcp_data/tf_cifar10_0_7960000038146973.hdf5'
    )

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32, 
          validation_split=0.2, 
          callbacks=[earlyStopping, mcp],
          verbose=1)
end_time = time.time() - start_time

# from keras.models import load_model
# model = load_model('mcp_data/tf_cifar10_0_7960000038146973.hdf5')
# mcp 모델을 로드할때 필요
# 불러올때는 데이터 셋과 아래의 평가, 예측 항목만 필요


# 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('걸린시간 : ', end_time)

# Epoch 00135: early stopping
# 313/313 [==============================] - 2s 6ms/step - loss: 0.6033 - accuracy: 0.7960
# loss :  0.6033015847206116
# acc :  0.7960000038146973
# 걸린시간 :  2349.84707570076