import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import regularizers

from data_factory import ds, dstep, validation_ds, vstep


log_dir="/home/ps/Projects/tensorflow-test/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 输入->卷积层1（5*5*32）->池化层->卷积层2（5*5*64）
# ->池化层->卷积层3（5*5*128）->池化层->全连接层（128）->Max(3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), bias_regularizer=regularizers.l2(1e-4), activation='relu', input_shape=(500, 500, 3)))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5), bias_regularizer=regularizers.l2(1e-4), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(128, kernel_size=(5,5), bias_regularizer=regularizers.l2(1e-4), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(128,
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5),
                activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(3, activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ds, epochs=500, steps_per_epoch=dstep,
          validation_data=validation_ds, validation_steps=vstep,
          callbacks=[tensorboard_callback])
