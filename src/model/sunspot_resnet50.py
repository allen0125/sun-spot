import sys, os
import datetime
sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import multi_gpu_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
import numpy as np

from data_factory import (
    ds, dstep, validation_ds, vstep,
    validation_all_ds
    # get_alldata_ds
)
# ds, dstep = get_alldata_ds("/home/ps/Projects/data/pre_strong/", 64)


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)


# 计算F1 Score
class Metrics(tf.keras.callbacks.Callback):
    """
    计算F1 Score Callback函数。

    """
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
        _val_precision = precision_score(
            total_targ, total_predict, average='macro')
        t = classification_report(
            total_targ, total_predict, target_names=["Alpha", "Beta", "BetaX"])
        print("\n")
        print(t)
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        return


# Tensorboard
log_dir="/home/ps/Projects/sunspot/logs/fit/" + datetime.datetime.now(
).strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# checkpoint settings
checkpoint_path = "checkpoint/07_22_resnet_addweight_09/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

base_model = ResNet50(
    include_top=False,
    weights="imagenet"
    )
# for layer in base_model.layers:
#     layer.trainable = False

layers_number = len(base_model.layers)

# resnet一共有175层layers
print("@@@@@@@@@layers number : ", layers_number)
top = int(layers_number * 0.8)
for layer in base_model.layers[:top]:
   layer.trainable = False
for layer in base_model.layers[top:]:
   layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# model.load_weights(checkpoint_path)

model.fit(ds, epochs=50, steps_per_epoch=dstep,
        # validation_data=validation_ds, validation_steps=vstep,
        callbacks=[tensorboard_callback,
                   Metrics(valid_data=validation_ds),
                   cp_callback,
                   reduce_lr,
                  ])
