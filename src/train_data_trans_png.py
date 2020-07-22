# Author: Allen Wang
# Date: 2020-06-19 13:57 Friday
# File: ~/Projects/tensorflow-test/src/data_utiles.py
# Description: 处理fits文件需要的脚本
# sys package

import os
import re
from multiprocessing import Process
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np
import cv2

from con_redis import SUN_REDIS

path ='/home/ps/Projects/data/trainset'
prepath='/home/ps/Projects/data/addweight_png'

Filelist=[]
typelist=['continuum','magnetogram']
labellist=['alpha','beta','betax']


def trans_fits(fits_path, save_path):
    c_hdu = fits.open(fits_path)
    c_hdu.verify('silentfix')
    image = c_hdu[1].data
    image_max = np.max(image)
    image_min = np.min(image)
    image_scalar = (image - image_min) / (image_max - image_min)
    plt.imsave(save_path, image_scalar)
    return image_scalar


def strong_png_betax(betax_image, save_file_name, t, l):

    betax_image_1 = cv2.flip(betax_image, 1, dst=None)
    pngname_1 = save_file_name + "_1.png"
    url_1 = prepath +'/'+t+'/'+l+'/'+ pngname_1
    plt.imsave(url_1, betax_image_1)

    betax_image_2 = cv2.flip(betax_image, -1, dst=None)
    pngname_2 = save_file_name + "_2.png"
    url_2 = prepath +'/'+t+'/'+l+'/'+ pngname_2
    plt.imsave(url_2, betax_image_2)


def strong_png_alpha(alpha_image, save_file_name, t, l):
    alpha_image_1 = cv2.flip(alpha_image, 1, dst=None)
    pngname_1 = save_file_name + "_1.png"
    url_1 = prepath+'/'+t+'/'+l+'/'+pngname_1
    plt.imsave(url_1, alpha_image_1)


def tras_to_png(t, l):
    Path=path+'/'+t+'/'+l
    i = 1
    for home, dirs, files in os.walk(Path):
        for filename in files:
            # 完成 magnetogram 对应文件名拼接
            mag_file_name = filename.replace('continuum', 'magnetogram')
            mag_file_path = path + '/magnetogram/' + l +'/'+  mag_file_name
            mag_png_name_list = re.split('[._]',mag_file_name)
            mag_png_name = '{}_{}_{}.{}'.format(
                mag_png_name_list[3], mag_png_name_list[4], 
                mag_png_name_list[5], mag_png_name_list[7]
            )
            m_pngname = mag_png_name + '.png'
            mag_png_path = prepath + '/magnetogram/' + l +'/'+ m_pngname

            #源fits文件全路径
            con_file_path = Path+'/'+filename
            namelist = re.split('[._]',filename)
            con_png_name = '{}_{}_{}.{}'.format(
                namelist[3], namelist[4], namelist[5],
                namelist[7]
            )
            # 处理后的文件名称及路径
            c_pngname = con_png_name+'.png'
            con_png_path = prepath+'/'+t+'/'+l+'/'+c_pngname

            # 处理fits文件
            # Filelist.append([pngname,ulr,t,l])
            c_image = trans_fits(con_file_path, con_png_path)
            m_image = trans_fits(mag_file_path, mag_png_path)

            """
            数据增强部分：BetaX类每一张图片都进行水平镜像和对角线镜像，分别包存为_1、_2
            Alpha类的每隔一张图片完成水平反转存为_1
            ！！！！ 已解决 一个BUG，没有注意需要白光Alpha增强和磁图Alpha增强图的对应。
            """
            if l == "betax":
                strong_png_betax(c_image, con_png_name, t, l)
                strong_png_betax(m_image, mag_png_name, 'magnetogram', l)

            # 需要保证alpha类增强的白光图和磁图是对应的
            if l == "alpha":
                if i % 2 == 0:
                    strong_png_alpha(c_image, con_png_name, t, l)
                    strong_png_alpha(m_image, mag_png_name, 'magnetogram', l)
                i += 1


def multi_trans_data():
    args_list = [
        ("continuum", "beta"),
        ("continuum", "betax"),
        ("continuum", "alpha"),
    ]
    process_list = list()

    for _args in args_list:  #开启3个子进程执行数据转换函数
        p = Process(target=tras_to_png,args=_args)  #实例化进程对象
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


if __name__ == '__main__':
    multi_trans_data()
