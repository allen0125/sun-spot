import os
import datetime
import pathlib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import regularizers
import numpy as np

# from data_factory import ds, dstep, validation_ds, vstep

def preprocess_image(image, image_size=500, channels=3):
    """
    按照image_size大小 resize image，将image正则化到[0, 1]区间
    return: 处理后的image图像
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image /= 255.0  # normalize to [0,1] range
    image = tf.reshape(image, [-1, image_size, image_size, 3])
    return image


def load_and_preprocess_image(path):
    """
    使用tensorflow io读取图片，调用preprocess_image处理并返回处理后的图片
    return: 返回 preprocess_image 处理后的图片
    """
    image = tf.io.read_file(path)
    # print(image)
    return preprocess_image(image)


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

checkpoint_path = "training_bigger_than_4_0702/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

data_root = pathlib.Path("/home/ps/Projects/data/test_png/continuum/")
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
print(all_image_paths)
with open('result.txt', 'a') as txt_file:
    for image_path in all_image_paths:
        image_data = load_and_preprocess_image(image_path)
        # print(type(image_data))
        result = model.predict(image_data)
        predict_index = np.argmax(result) + 1
        image_id = image_path.split("/")[-1].split(".")[0].split("_")[-1]
        line = image_id + ' ' + str(predict_index) + '\n'
        print(line)
        txt_file.write(line)
