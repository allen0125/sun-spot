import pathlib
import random
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import datetime


log_dir="/home/ps/Projects/tensorflow-test/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

train_data_root_orig = '/home/ps/Projects/tensorflow-test/data/test/'
test_data_root_orig = '/home/ps/Projects/tensorflow-test/data/train/'

def get_ds(data_root_orig):
    data_root = pathlib.Path(data_root_orig)
    print(data_root)

    for item in data_root.iterdir():
        print(item)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    print(image_count)

    print(all_image_paths[:10])

    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    print(label_names)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print(label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    img_path = all_image_paths[0]
    print(img_path)
    img_raw = tf.io.read_file(img_path)
    print(repr(img_raw)[:100]+"...")
    img_tensor = tf.image.decode_image(img_raw)

    print(img_tensor.shape)
    print(img_tensor.dtype)

    img_final = tf.image.resize(img_tensor, [192, 192])
    img_final = img_final/255.0
    print(img_final.shape)
    print(img_final.numpy().min())
    print(img_final.numpy().max())

    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [192, 192])
        image /= 255.0  # normalize to [0,1] range

        return image


    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    print(path_ds)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    for label in label_ds.take(10):
        print(label_names[label.numpy()])
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    print(image_label_ds)

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

    # 元组被解压缩到映射函数的位置参数中
    def load_and_preprocess_from_path_label(path, label):
        return load_and_preprocess_image(path), label

    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    print(image_label_ds)


    BATCH_SIZE = 32

    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    # ds = image_label_ds.shuffle(buffer_size=image_count)
    # ds = ds.repeat()
    # ds = ds.batch(BATCH_SIZE)
    # # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    # ds = ds.prefetch(buffer_size=AUTOTUNE)
    # print(ds)
    ds = image_label_ds.cache()
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return ds

# ds = get_ds(train_data_root_orig)
validation_ds = get_ds(test_data_root_orig)

# mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
# mobile_net.trainable=True

# def change_range(image,label):
#     return 2*image-1, label

# keras_ds = ds.map(change_range)
# keras_val_ds = validation_ds.map(change_range)
# image_batch, label_batch = next(iter(keras_ds))
# feature_map_batch = mobile_net(image_batch)
# print(feature_map_batch.shape)

# model = tf.keras.Sequential([
#     mobile_net,
#     tf.keras.layers.GlobalAveragePooling2D(),
#     # tf.keras.layers.Dense(128, activation = 'softmax'),
#     # tf.keras.layers.Dropout(0.5),
#     # tf.keras.layers.Dense(64, activation = 'softmax'),
#     # tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(3, activation = 'softmax')])

# logit_batch = model(image_batch).numpy()

# print("min logit:", logit_batch.min())
# print("max logit:", logit_batch.max())

# print("Shape:", logit_batch.shape)


# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss='sparse_categorical_crossentropy',
#               metrics=["accuracy"])
# len(model.trainable_variables)
# model.summary()

# # steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
# # print(steps_per_epoch)

# model.fit(ds, epochs=500, steps_per_epoch=500,
#           validation_data=validation_ds, validation_steps=100,
#           callbacks=[tensorboard_callback])






# 在开始计时之前
# 取得单个 batch 来填充 pipeline（管道）（填充随机缓冲区）
it = iter(validation_ds.take(validation_steps))
next(it)

for i,(images,labels) in enumerate(it):
    print(images, labels)
    print("-----------------------------")
