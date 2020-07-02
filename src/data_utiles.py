# Author: Allen Wang
# Date: 2020-06-19 13:57 Friday
# File: ~/Projects/tensorflow-test/src/data_utiles.py
# Description: 用于测试Tensorflow OpenCV 处理数据的脚本
# sys package
# import os
# from multiprocessing import Process
# import unittest

# # thread package
# import matplotlib
# from PIL import Image
# import cv2
# import tensorflow as tf
# from astropy.io import fits
# import numpy as np


# DATA_DIR = '/home/ps/Projects/data/trainset/'
# TEST_FILE_DIR = '/home/ps/Projects/data/trainset/continuum/alpha/\
# hmi.sharp_720s.5818.20150802_044800_TAI.continuum.fits'


# def fits_to_jpg(file_dir):
#     test_file = fits.open(file_dir)
#     test_file.verify('fix')
#     data = test_file[1].data
#     data_max = np.amax(data)
#     data_min = np.amin(data)
#     data_grey = 255 * (data - data_min) / (data_max - data_min)
#     data_grey_r = data_grey.astype(float)
#     file_name_list = 'hmi.sharp_720s.4166.20140526_191200_TAI.continuum.fits'.split('.')
#     save_images = Image.fromarray(data_grey_r)
#     if save_images.mode != 'RGB':
#         save_images = save_images.convert('RGB')
#     # save_images.save("4166.jpg")
#     return save_images

# continuum_beta = list()
# continuum_betax = list()
# continuum_alpha = list()

# magnetogram_beta = list()
# magnetogram_betax = list()
# magnetogram_alpha = list()


# def trans_data_to_gray(data_dir, cam_class, fits_class):

#     data_list = list()

#     for root, dirs, files in os.walk(data_dir):
#         print("root", root)  # 当前目录路径
#         print("dirs", dirs)  # 当前路径下所有子目录
#         # print("files", files)  # 当前路径下所有非目录子文件
#         if root.endswith(cam_class + "/" + fits_class):
#             name_list = list()
#             for fits_file in files:
#                 file_name_split = fits_file.split('.')
#                 point_and_time = file_name_split[2] + "/" + file_name_split[3]
#                 if point_and_time in name_list:
#                     continue
#                 name_list.append(point_and_time)
#                 image = fits_to_jpg(root + "/" + fits_file)
#                 data_list.append((point_and_time, np.array(image)))
#     data_list = np.array(data_list)
#     np.save('/home/ps/Projects/tensorflow-test/data/{}-{}.npy'.format(cam_class, fits_class), data_list)
#     print(len(data_list))


# def test_trans_data():
#     args_list = [
#         (DATA_DIR, "continuum", "beta"),
#         (DATA_DIR, "continuum", "betax"),
#         (DATA_DIR, "continuum", "alpha"),
#         (DATA_DIR, "magnetogram", "beta"),
#         (DATA_DIR, "magnetogram", "betax"),
#         (DATA_DIR, "magnetogram", "alpha"),
#     ]
#     process_list = list()

#     for _args in args_list:  #开启6个子进程执行数据转换函数
#         p = Process(target=trans_data_to_gray,args=_args)  #实例化进程对象
#         p.start()
#         process_list.append(p)

#     for p in process_list:
#         p.join()

#     # trans_data_to_gray(DATA_DIR)


# def fits_to_png(file_dir):
#     test_file = fits.open(file_dir)
#     test_file.verify('fix')
#     data = test_file[1].data
#     print(data.shape)
#     # data_max = np.amax(data)
#     # data_min = np.amin(data)
#     # data_grey = 255 * (data - data_min) / (data_max - data_min)
#     # data_grey_r = data_grey.astype(float)
#     # file_name_list = 'hmi.sharp_720s.4166.20140526_191200_TAI.continuum.fits'.split('.')
#     # save_images = Image.fromarray(data_grey_r)
#     # if save_images.mode != 'RGB':
#     #     save_images = save_images.convert('RGB')
#     # # save_images.save("4166.jpg")
#     # return save_images


# if __name__ == '__main__':
#     # test_trans_data()
#     fits_to_png(TEST_FILE_DIR)




import os
import re
from multiprocessing import Process
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np

path ='/home/ps/Projects/data/test_input/'
prepath='/home/ps/Projects/data/test_png/'

Filelist=[]
typelist=['continuum','magnetogram']
labellist=['alpha','beta','betax']


def tras_to_png(t, l):
    Path=path+'/'+t+'/'+l
    for home, dirs, files in os.walk(Path):
        for filename in files:
            #源fits文件全路径
            name=Path+'/'+filename
            namelist=re.split('[._]',filename)
            #处理后的文件名称及路径
            pngname=namelist[3]+'_'+namelist[4]+'_'+namelist[5]+'.'+namelist[7]+'.'+'png'
            ulr=prepath+'/'+t+'/'+l+'/'+pngname
            hdu=fits.open(name)
            hdu.verify('fix')
            image = hdu[1].data
            Imax=np.max(image)
            Imin=np.min(image)
            Image_Scalar=(image-Imin)/(Imax-Imin)
            plt.imsave(ulr,Image_Scalar)
            Filelist.append([pngname,ulr,t,l])


def multi_trans_data():
    args_list = [
        ("continuum", "beta"),
        ("continuum", "betax"),
        ("continuum", "alpha"),
        ("magnetogram", "beta"),
        ("magnetogram", "betax"),
        ("magnetogram", "alpha"),
    ]
    process_list = list()

    for _args in args_list:  #开启6个子进程执行数据转换函数
        p = Process(target=tras_to_png,args=_args)  #实例化进程对象
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


if __name__ == '__main__':
    multi_trans_data()
    print("test")
