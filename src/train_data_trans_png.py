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

path ='/home/ps/Projects/data/trainset/'
prepath='/home/ps/Projects/data/pre'

Filelist=[]
typelist=['continuum','magnetogram']
labellist=['alpha','beta','betax']


def tras_to_png(t, l):
    Path=path+'/'+t+'/'+l
    i = 1
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

            """
            数据增强部分：BetaX类每一张图片都进行水平镜像和对角线镜像，分别包存为_1、_2
            Alpha类的每隔一张图片完成水平反转存为_1
            ！！！！ 有一个BUG，没有注意需要白光Alpha增强和磁图Alpha增强图的对应。
            """
            if l == "betax":
                Image_Scalar_1 = cv2.flip(Image_Scalar, 1, dst=None)
                pngname_1=namelist[3]+'_'+namelist[4]+'_'+namelist[5]+'.'+namelist[7]+ "_1" + '.'+'png'
                ulr_1=prepath+'/'+t+'/'+l+'/'+pngname_1
                plt.imsave(ulr_1,Image_Scalar_1)
                Image_Scalar_2 = cv2.flip(Image_Scalar, -1, dst=None)
                pngname_2=namelist[3]+'_'+namelist[4]+'_'+namelist[5]+'.'+namelist[7]+ "_2" + '.'+'png'
                ulr_2=prepath+'/'+t+'/'+l+'/'+pngname_2
                plt.imsave(ulr_2, Image_Scalar_2)
            if l == "alpha":
                if i % 2 == 0:
                    Image_Scalar_1 = cv2.flip(Image_Scalar, 1, dst=None)
                    pngname_1=namelist[3]+'_'+namelist[4]+'_'+namelist[5]+'.'+namelist[7]+ "_1" + '.'+'png'
                    ulr_1=prepath+'/'+t+'/'+l+'/'+pngname_1
                    plt.imsave(ulr_1,Image_Scalar_1)
                i += 1


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
