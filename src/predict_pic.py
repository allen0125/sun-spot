import os
import datetime
import pathlib
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np

# from data_factory import ds, dstep, validation_ds, vstep

def preprocess_image(image, image_size=224, channels=3):
    """
    按照image_size大小 resize image，将image正则化到[0, 1]区间
    return: 处理后的image图像
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    # image /= 255.0  # normalize to [0,1] range
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


base_model = ResNet50(
    include_top=False,
    weights="imagenet"
    )

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_path = "checkpoint/07_22_resnet_addweight_0{}/cp.ckpt".format(
    sys.argv[1])
model.load_weights(checkpoint_path)

data_root = pathlib.Path(
    "/home/ps/Projects/data/test_addweight/test_addweight_c0.{}_png/".format(
        sys.argv[1]))
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
print(all_image_paths)
with open('result_0722_c0.{}.txt'.format(sys.argv[1]), 'a') as txt_file:
    for image_path in all_image_paths:
        image_data = load_and_preprocess_image(image_path)
        # print(type(image_data))
        result = model.predict(image_data)
        predict_index = np.argmax(result) + 1
        image_id = image_path.split("/")[-1].split(".")[0].split("_")[-1]
        line = image_id + ' ' + str(predict_index) + '\n'
        print(line)
        txt_file.write(line)
