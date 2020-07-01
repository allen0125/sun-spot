from tensorflow import keras
from tensorflow.keras.preprocessing.image import image_dataset_from_directory

train_ds = image_dataset_from_directory(
    directory='/home/ps/Projects/tensorflow-test/data/train/',
    labels='inferred',
    label_mode='integers',
    batch_size=128,
    image_size=(256, 256))
validation_ds = image_dataset_from_directory(
    directory='/home/ps/Projects/tensorflow-test/data/test/',
    labels='inferred',
    label_mode='integers',
    batch_size=128,
    image_size=(256, 256))

model = keras.applications.Xception(weights=None, input_shape=(256, 256, 3), classes=3)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(train_ds, epochs=10, validation_data=validation_ds)
