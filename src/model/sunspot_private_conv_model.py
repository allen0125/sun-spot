import os
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
import numpy as np

from data_factory import ds, dstep, validation_ds, vstep, validation_all_ds

checkpoint_path = "training_bigger_than_4_0702_all_data/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


def Get_Average(list):
    sum = 0
    for item in list:     
        sum += item  
    return sum/len(list)


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        it = iter(self.validation_data.take(vstep))
        next(it)
        total_predict = np.array([])
        total_targ = np.array([])

        for i,(images,labels) in enumerate(it):
            val_predict = np.argmax(self.model.predict(images), -1)
            val_targ = labels.numpy()
            if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
                val_targ = np.argmax(val_targ, -1)
            total_predict = np.concatenate((total_predict, val_predict),axis=0)
            total_targ = np.concatenate((total_targ, val_targ),axis=0)

        _val_f1 = f1_score(total_targ, total_predict, average='macro')
        _val_recall = recall_score(total_targ, total_predict, average='macro')
        _val_precision = precision_score(total_targ, total_predict, average='macro')
        t = classification_report(total_targ, total_predict, target_names=["Alpha", "Beta", "BetaX"])
        print("\n")
        print(t)
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        return


log_dir="/home/ps/Projects/sunspot/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
          validation_data=validation_ds, 
          callbacks=[tensorboard_callback,
                     Metrics(valid_data=validation_ds),
                    #  cp_callback,
                     ])
