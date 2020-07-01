import numpy as np
import tensorflow as tf
from PIL import Image
import random
import pylab as pl
import matplotlib


DATASET_BASE_DIR = "/home/ps/Projects/tensorflow-test/data/"
save_dir = "/home/ps/Projects/tensorflow-test/data"

args_list = [
        ("continuum", "beta"),
        ("continuum", "betax"),
        ("continuum", "alpha"),
        ("magnetogram", "beta"),
        ("magnetogram", "betax"),
        ("magnetogram", "alpha"),
    ]

continuum_beta_list = list()
continuum_betax_list = list()
continuum_alpha_list = list()

magnetogram_beta_list = list()
magnetogram_betax_list = list()
magnetogram_alpha_list = list()

continuum_beta = np.load(
    DATASET_BASE_DIR + '{}-{}.npy'.format("continuum", "beta"), allow_pickle=True)

continuum_betax = np.load(
    DATASET_BASE_DIR + '{}-{}.npy'.format("continuum", "betax"), allow_pickle=True)

continuum_alpha = np.load(
    DATASET_BASE_DIR + '{}-{}.npy'.format("continuum", "alpha"), allow_pickle=True)

magnetogram_beta = np.load(
    DATASET_BASE_DIR + '{}-{}.npy'.format("magnetogram", "beta"), allow_pickle=True)

magnetogram_betax = np.load(
    DATASET_BASE_DIR + '{}-{}.npy'.format("magnetogram", "betax"), allow_pickle=True)

magnetogram_alpha = np.load(
    DATASET_BASE_DIR + '{}-{}.npy'.format("magnetogram", "alpha"), allow_pickle=True)


print(len(magnetogram_beta))
print(len(magnetogram_betax))
print(len(magnetogram_alpha))


def build_point_list(dataset, point_list):
    for name, _ in dataset:
        watch_point = name.split('/')[0]
        if watch_point not in point_list:
            point_list.append(watch_point)


def build_dataset(dataset, point_list, label, train_x, train_y, test_x, test_y):
    """
    return train_x, train_y, test_x, test_y

    """
    test_point = dict()
    train_point = dict()

    point_watch_day = {}
    for name, image in dataset:
        watch_point, watch_date = name.split('/')
        watch_date = watch_date.split('_')[0]
        print(watch_point)
        if watch_point not in point_watch_day.keys():
            point_watch_day[watch_point] = [watch_date]
        else:
            point_watch_day[watch_point].append(watch_date)

    print(point_watch_day)

    # 根据每个点观测的天数划分训练集及测试集
    for point in point_watch_day.keys():
        if len(point_watch_day[point]) > 9:
            test_point[point] = point_watch_day[point][1::4]
            for day in point_watch_day[point]:
                if day not in test_point[point]:
                    if point not in train_point.keys():
                        train_point[point] = [day]
                    else:
                        train_point[point].append(day)

    for name, image in dataset:
        watch_point, watch_date = name.split('/')
        watch_date = watch_date.split('_')[0]
        if watch_point in test_point.keys() and watch_date in test_point[watch_point]:
            test_x.append(image)
            test_y.append(label)
            pic = Image.fromarray(image)
            name = name.replace('/', '_')
            pic.save(
                save_dir + '/' + 'test/{}/{}.jpg'.format(
                    "".join(str(i) for i in label), name))
        elif watch_point in train_point.keys() and watch_date in train_point[watch_point]:
            train_x.append(image)
            train_y.append(label)
            name = name.replace('/', '_')
            pic = Image.fromarray(image)
            pic.save(
                save_dir + '/' + 'train/{}/{}.jpg'.format(
                    "".join(str(i) for i in label), name))

    # 查看每个出现次数频率，例如观察了45天的点有5个
    # total_data_status = dict()
    # for key in point_watch_day.keys():
    #     if len(point_watch_day[key]) not in total_data_status:
    #         total_data_status[len(point_watch_day[key])] = 1
    #     else:
    #         total_data_status[len(point_watch_day[key])] += 1
    
    # print(total_data_status)
    # x = list(total_data_status.keys())
    # y = list()
    # for i in x:
    #     y.append(total_data_status[i])
    # print(x)
    # print(y)

    return train_x, train_y, test_x, test_y


def make_train_dataset():
    train_x = list()
    train_y = list()
    test_x = list()
    test_y = list() 

    build_point_list(continuum_beta, continuum_beta_list)
    build_point_list(continuum_betax, continuum_betax_list)
    build_point_list(continuum_alpha, continuum_alpha_list)

    # build_point_list(magnetogram_beta, magnetogram_beta_list)
    # build_point_list(magnetogram_betax, magnetogram_betax_list)
    # build_point_list(magnetogram_alpha, magnetogram_alpha_list)

    build_dataset(continuum_beta, continuum_beta_list,
                  [0,0,1], train_x, train_y, test_x, test_y)

    build_dataset(continuum_betax, continuum_betax_list,
                  [0,1,0], train_x, train_y, test_x, test_y)

    build_dataset(continuum_alpha, continuum_alpha_list,
                  [1,0,0], train_x, train_y, test_x, test_y)
    
    # build_dataset(magnetogram_beta, magnetogram_beta_list,
    #               [0,0,1], train_x, train_y, test_x, test_y)
    
    # build_dataset(magnetogram_betax, magnetogram_betax_list,
    #               [0,1,0], train_x, train_y, test_x, test_y)

    # build_dataset(magnetogram_alpha, magnetogram_alpha_list,
    #               [1,0,0], train_x, train_y, test_x, test_y)

    return train_x, train_y, test_x, test_y

make_train_dataset()


# def save_dataset_pic(save_dir):
#     train_x, train_y, test_x, test_y = make_train_dataset()
#     for pic_index, pic in enumerate(train_x):
#         pic = Image.fromarray(pic)
#         pic.save(
#             save_dir + '/' + 'train/{}/{}.jpg'.format(
#                 "".join(str(i) for i in train_y[pic_index]), pic_index))
    
#     for pic_index, pic in enumerate(test_x):
#         pic = Image.fromarray(pic)
#         pic.save(
#             save_dir + '/' + 'test/{}/{}.jpg'.format(
#                 "".join(str(i) for i in test_y[pic_index]), pic_index))


# save_dataset_pic("/home/ps/Projects/tensorflow-test/data")
