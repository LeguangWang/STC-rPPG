import h5py
import numpy as np


f = h5py.File('PURE/Rnet_PURE_test.hdf5','r') #打开h5文件
i=0
# 可以查看所有的主键
for key in f.keys():

 print(i,f[key].name)
 # a = f['preprocessed_label']
 print("")
 i=i+1
# "subject10
# /subject15
# /subject18
# /subject34
# /subject36
# /subject39
# /subject44
# /subject47
# /subject49"