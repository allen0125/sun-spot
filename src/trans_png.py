import os
import re
from multiprocessing import Process
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np

path ='/home/ps/Projects/data/test_input'
prepath='/home/ps/Projects/data/test_png/'

Filelist=[]
typelist=['continuum','magnetogram']
labellist=['alpha','beta','betax']


def tras_to_png(t, l):
    Path=path+'/'+t
    print(Path)
    for home, dirs, files in os.walk(Path):
        for filename in files:
            #源fits文件全路径
            name=Path+'/'+filename
            namelist=re.split('[._]',filename)
            #处理后的文件名称及路径
            pngname=namelist[3]+'_'+namelist[4]+'_'+namelist[5]+'.'+namelist[7]+'.'+'png'
            ulr=prepath+t+'/'+pngname
            hdu=fits.open(name)
            hdu.verify('fix')
            image = hdu[1].data
            Imax=np.max(image)
            Imin=np.min(image)
            Image_Scalar=(image-Imin)/(Imax-Imin)
            plt.imsave(ulr,Image_Scalar)
            Filelist.append([pngname,ulr,t])


def multi_trans_data():
    print("1")
    args_list = [
        ("continuum", "beta"),
        ("magnetogram", "beta")
    ]
    print(args_list)
    process_list = list()

    for _args in args_list:
        p = Process(target=tras_to_png,args=_args)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


if __name__ == '__main__':
    multi_trans_data()