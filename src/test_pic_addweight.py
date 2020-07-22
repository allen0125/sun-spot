# Author: Allen Wang
# Date: 2020-07-22 08:04 Wednesday
# File: ~/Projects/sunspot/src/addweight.py
# Description: add con mag pic weight

import os
import re
import sys

from multiprocessing import Process
from matplotlib import pyplot as plt
from PIL import Image
from astropy.io import fits
import numpy as np
import cv2

from con_redis import SUN_REDIS

weight_alpha = sys.argv[1]
print(weight_alpha)

path ='/home/ps/Projects/data/test_png'
prepath='/home/ps/Projects/data/test_addweight/test_addweight_c{}_png'.format(
    weight_alpha
)

Filelist=[]
typelist=['continuum','magnetogram']
labellist=['alpha','beta','betax']


def add_image(img1_path, img2_path, alpha):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    h, w, _ = img1.shape
    # 函数要求两张图必须是同一个size
    img2 = cv2.resize(img2, (w,h), interpolation=cv2.INTER_AREA)
    #alpha，beta，gamma可调
    beta = 1 - float(weight_alpha)
    gamma = 0
    img_add = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    return img_add


def tras_to_png(t, l):
    Path=path+'/'+t+'/'+l
    i = 1
    for home, dirs, files in os.walk(Path):
        for filename in files:
            con_file_path = Path+'/'+filename
            mag_file_path = con_file_path.replace('continuum', 'magnetogram')
            addweight_img = add_image(con_file_path, mag_file_path, float(weight_alpha))

            addweight_img_name = filename.replace('continuum', 'addweight')
            addweight_img_path = '{}/{}/{}.png'.format(
                prepath, l, addweight_img_name
            )


            # plt.imsave(addweight_img_path, image)
            cv2.imwrite(
                addweight_img_path, addweight_img,)


def multi_trans_data():
    args_list = [
        ("continuum", "000"),
        # ("continuum", "betax"),
        # ("continuum", "alpha"),
    ]
    process_list = list()

    for _args in args_list:  #开启3个子进程执行数据转换函数
        p = Process(target=tras_to_png, args=_args)  #实例化进程对象
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


if __name__ == '__main__':
    multi_trans_data()
