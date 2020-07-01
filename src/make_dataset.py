import numpy as np
import tensorflow as tf
from PIL import Image
import random
import pylab as pl
import matplotlib


DATASET_BASE_DIR = "/home/ps/Projects/tensorflow-test/data/"

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
    test_point = random.sample(point_list,int(len(point_list) * 0.2))
    train_point = [i for i in point_list if i not in test_point]

    print(label)
    print(test_point)
    print(train_point)
    print("-------------")

    point_watch_day = {}
    for name, image in dataset:
        watch_point, watch_date = name.split('/')
        watch_date = watch_date.split('_')[0]
        print(watch_point)
        print(watch_date.split('_')[0])
        if watch_point not in point_watch_day.keys():
            point_watch_day[watch_point] = [watch_date]
        else:
            point_watch_day[watch_point].append(watch_date)

        if watch_point in test_point:
            test_x.append(image)
            test_y.append(label)
        elif watch_point in train_point:
            train_x.append(image)
            train_y.append(label)

    return train_x, train_y, test_x, test_y


def make_train_dataset():
    train_x = list()
    train_y = list()
    test_x = list()
    test_y = list() 

    build_point_list(continuum_beta, continuum_beta_list)
    build_point_list(continuum_betax, continuum_betax_list)
    build_point_list(continuum_alpha, continuum_alpha_list)

    build_point_list(magnetogram_beta, magnetogram_beta_list)
    build_point_list(magnetogram_betax, magnetogram_betax_list)
    build_point_list(magnetogram_alpha, magnetogram_alpha_list)

    build_dataset(continuum_beta, continuum_beta_list,
                  [0,0,1], train_x, train_y, test_x, test_y)

    build_dataset(continuum_betax, continuum_betax_list,
                  [0,1,0], train_x, train_y, test_x, test_y)

    build_dataset(continuum_alpha, continuum_alpha_list,
                  [1,0,0], train_x, train_y, test_x, test_y)
    
    build_dataset(magnetogram_beta, magnetogram_beta_list,
                  [0,0,1], train_x, train_y, test_x, test_y)
    
    build_dataset(magnetogram_betax, magnetogram_betax_list,
                  [0,1,0], train_x, train_y, test_x, test_y)

    build_dataset(magnetogram_alpha, magnetogram_alpha_list,
                  [1,0,0], train_x, train_y, test_x, test_y)

    return train_x, train_y, test_x, test_y

# make_train_dataset()


def save_dataset_pic(save_dir):
    train_x, train_y, test_x, test_y = make_train_dataset()
    for pic_index, pic in enumerate(train_x):
        pic = Image.fromarray(pic)
        pic.save(
            save_dir + '/' + 'train/{}/{}.jpg'.format(
                "".join(str(i) for i in train_y[pic_index]), pic_index))
    
    for pic_index, pic in enumerate(test_x):
        pic = Image.fromarray(pic)
        pic.save(
            save_dir + '/' + 'test/{}/{}.jpg'.format(
                "".join(str(i) for i in test_y[pic_index]), pic_index))


save_dataset_pic("/home/ps/Projects/tensorflow-test/data")

