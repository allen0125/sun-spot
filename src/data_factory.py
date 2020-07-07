import pathlib
import random
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import datetime


# 训练集及测试集地址

train_data_root_orig = '/home/ps/Projects/data/pre/continuum/'
test_data_root_orig = '/home/ps/Projects/data/pre/continuum/'


def preprocess_image(image, image_size=331, channels=3):
    """
    按照image_size大小 resize image，将image正则化到[0, 1]区间
    return: 处理后的image图像
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    """
    使用tensorflow io读取图片，调用preprocess_image处理并返回处理后的图片
    return: 返回 preprocess_image 处理后的图片
    """
    image = tf.io.read_file(path)
    return preprocess_image(image)


def get_image_paths_labels(data_root_orig):
    """
    完成所有图片数据路径集合计算
    完成所有图片数据对应标签计算
    形成两个 all_image_paths, all_image_labels

    """
    data_root = pathlib.Path(data_root_orig)
    print("root_is: ", data_root)

    # 获取所有图片路径及对应标签
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    label_names = sorted(
        item.name for item in data_root.glob('*/') if item.is_dir())
    print("label names: ", label_names)
    label_to_index = dict(
        (name, index) for index, name in enumerate(label_names))
    print("!!!!!!!!!!!!!*************** Label Index is : ", label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    
    return all_image_paths, all_image_labels


def split_train_test_dataset(all_image_paths, all_image_labels):
    train_paths = list()
    train_labels = list()
    test_paths = list()
    test_labels = list()

    points_watch_days = dict()

    for image_index, image_path in enumerate(all_image_paths):
        image_name = pathlib.Path(image_path).name
        point_info = image_name.split('.')[0]
        point_name, point_watch_day, _ = point_info.split('_')
        if point_name not in points_watch_days.keys():
            points_watch_days[point_name] = [point_watch_day]
        else:
            if point_watch_day not in points_watch_days[point_name]:
                points_watch_days[point_name].append(point_watch_day)
    
    train_points_days = dict()
    test_points_days = dict()

    less_points_temp_list = list()
    
    for key in points_watch_days.keys():
        points_watch_days[key].sort()
        # 划分观测数据大于4天的数据
        if len(points_watch_days[key]) > 4:
            test_days = points_watch_days[key][1::3]
            train_days = [
                point for point in points_watch_days[
                    key] if point not in test_days]
            train_points_days[key] = train_days
            test_points_days[key] = test_days
        # 完成观测天数少于4天的数据划分
        else:
            less_points_temp_list.append(key)

    less_test_points = random.sample(
        less_points_temp_list, int(len(less_points_temp_list) * 0.3))
    print("less test points num: ", len(less_test_points))
    print("less points num", len(less_points_temp_list))
    for key in less_test_points:
        test_points_days[key] = points_watch_days[key]
    for key in less_points_temp_list:
        if key not in less_test_points:
            train_days = points_watch_days[key]
            train_points_days[key] = train_days

    for image_index, image_path in enumerate(all_image_paths):
        image_name = pathlib.Path(image_path).name
        point_info = image_name.split('.')[0]
        point_name, point_watch_day, _ = point_info.split('_')
        if point_name in train_points_days.keys() and \
        point_watch_day in train_points_days[point_name]:
            train_paths.append(image_path)
            train_labels.append(all_image_labels[image_index])
        if point_name in test_points_days.keys() and \
        point_watch_day in test_points_days[point_name]:
            test_paths.append(image_path)
            test_labels.append(all_image_labels[image_index])

    return train_paths, train_labels, test_paths, test_labels


def make_dataset(image_paths, image_labels, batch_size=32):
    """
    完成ds计算
    """
    BATCH_SIZE = batch_size

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    print(path_ds)
    image_ds = path_ds.map(
        load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据
    # 被充分打乱。
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    image_count = len(image_paths)
    ds = image_label_ds.cache()
    ds = ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    steps_per_epoch=tf.math.ceil(image_count/BATCH_SIZE).numpy()
    return ds, steps_per_epoch, image_label_ds


def get_ds(data_root_orig, batch_size=32):
    """
    输入图片地址，batch size，返回训练与验证dataset数据集

    """
    
    
    all_image_paths, all_image_labels = get_image_paths_labels(data_root_orig)

    # 计算所有image 数量
    image_count = len(all_image_paths)
    print("total image count: ", image_count)
    train_paths, train_labels, \
    test_paths, test_labels = split_train_test_dataset(
        all_image_paths, all_image_labels)
    print("训练数据量", len(train_paths))
    print("验证数据量", len(test_paths))
    print("训练Alpha：", train_labels.count(0))
    print("训练Beta：", train_labels.count(1))
    print("训练Betax：", train_labels.count(2))
    print("测试Alpha：", test_labels.count(0))
    print("测试Beta：", test_labels.count(1))
    print("测试Betax：", test_labels.count(2))
    ds, dstep, _ = make_dataset(train_paths, train_labels, batch_size)
    validation_ds, vstep, validation_all_ds = make_dataset(
        test_paths, test_labels, batch_size)
    return ds, dstep, validation_ds, vstep, validation_all_ds


def get_alldata_ds(data_root_orig, batch_size=32):
    """
    输入图片地址，batch size，返回所有数据dataset数据集

    """

    all_image_paths, all_image_labels = get_image_paths_labels(data_root_orig)

    # 计算所有image 数量
    image_count = len(all_image_paths)
    print("total image count: ", image_count)
    train_paths, train_labels, \
    test_paths, test_labels = split_train_test_dataset(
        all_image_paths, all_image_labels)
    print("训练数据量", len(train_paths))
    print("验证数据量", len(test_paths))
    print("训练Alpha：", train_labels.count(0))
    print("训练Beta：", train_labels.count(1))
    print("训练Betax：", train_labels.count(2))
    ds, dstep, _ = make_dataset(all_image_paths, all_image_labels, batch_size)
    return ds, dstep

with tf.device('/device:GPU:0'):
    ds, dstep, validation_ds, vstep, validation_all_ds = get_ds(
        train_data_root_orig)
