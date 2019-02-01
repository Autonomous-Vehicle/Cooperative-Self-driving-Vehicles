import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import cv2
import csv
import matplotlib
import keras
import numpy as np
import os
import random
import keras.backend as K
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.switch_backend('agg')
matplotlib.use('Agg')


def val_output(s):
    error1 = s[0]
    error2 = s[1]
    error3 = s[2]
    error4 = s[3]
    return error1,error2,error3,error4

pred = []
real = []
df = pd.read_csv("modelsp.csv", dtype={'GT': np.float, 'A': np.float, 'E': np.float, 'D': np.float, 'F': np.float})
#df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

#fnames = np.array('Ch2/' + df['filename'])
GT = np.array(df['GT'])

plt.hist(GT, normed=True, bins=1000,cumulative=True)
plt.ylabel('Distribution');
plt.savefig ('Probabilitynorm.png')
plt.hist(GT,bins=1000,linewidth=4.0 , color='b')
plt.ylim([-2, 850])
plt.ylabel('Distribution');
plt.savefig ('ProbabilityNOtnorm.png')
#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)
len_x=len(GT)
t = np.arange(0, len_x,1)
one_array=np.ones(len_x)
line=np.zeros(len_x)

min_GT=min(GT)
max_GT=max(GT)
print(min_GT)
print(max_GT)
max_plotGT=max_GT*one_array
min_plotGT=min_GT*one_array

plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([-2.5, 2.5])
plt.xlim([0, len_x])
plt.plot(GT,'b', label='GT')
plt.plot(t,line,'r',t,max_plotGT,'r',t,min_plotGT,'r',linewidth=2.0, label='Error')
#plt.xlabel('xlabel', fontsize=18)
#plt.ylabel('ylabel', fontsize=16)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('GT.png')


A = np.array(df['A'])

#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)
plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([-4, 4])
plt.xlim([0, len_x])
#plt.xticks([0.4,0.14,0.2,0.2], fontsize = 50)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
plt.plot(A,'b', label='A')
#plt.ylabel('ylabel', fontsize=16)
#plt.legend()
plt.savefig ('A.png')



errorA=GT-A
min_A=min(errorA)
max_A=max(errorA)
max_plot=max_A*one_array
min_plot=min_A*one_array


plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([-1.5, 1.5])
plt.xlim([0, len_x])
plt.plot(t,errorA,'b', label='Error')
plt.plot(t,line,'r',t,max_plot,'r',t,min_plot,'r',linewidth=2.0, label='Error')
#plt.xlabel('xlabel', fontsize=18)
#plt.ylabel('ylabel', fontsize=16)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('errorA.png')


D = np.array(df['D'])
#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)

plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([-4, 4])
plt.xlim([0, len_x])
plt.plot(D,'b', label='F')
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('D.png')

errorD=GT-D
min_A=min(errorD)
max_A=max(errorD)
max_plot=max_A*one_array
min_plot=min_A*one_array

plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([-1.5, 1.5])
plt.xlim([0, len_x])
plt.plot(errorD,'b',  label='Error')
plt.plot(t,line,'r',t,max_plot,'r',t,min_plot,'r',linewidth=2.0, label='Error')
#plt.xlabel('xlabel', fontsize=18)
#plt.ylabel('ylabel', fontsize=16)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('errorD.png')

E = np.array(df['E'])
#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)
maximun=max(E)
minimun=min(E)
plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([minimun-0.3, maximun+0.3])
plt.xlim([0, len_x])
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
plt.plot(E,'b', label='E')
#plt.legend()
plt.savefig ('E.png')

errorE=GT-E
min_A=min(errorE)
max_A=max(errorE)
max_plot=max_A*one_array
min_plot=min_A*one_array


maximun=max(errorE)
minimun=min(errorE)
plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([minimun-0.3, maximun+0.3])
plt.xlim([0, len_x])
plt.plot(errorE,'b',  label='Error')
plt.plot(t,line,'r',t,max_plot,'r',t,min_plot,'r',linewidth=2.0, label='Error')
#plt.xlabel('xlabel', fontsize=18)
#plt.ylabel('ylabel', fontsize=16)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('errorE.png')

F = np.array(df['F'])
#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)

plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([-4, 4])
plt.xlim([0, len_x])
plt.plot(F,'b', label='F')
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('F.png')

errorF=GT-F
min_A=min(errorF)
max_A=max(errorF)
max_plot=max_A*one_array
min_plot=min_A*one_array


plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([-1.5, 1.5])
plt.xlim([0, len_x])
plt.plot(errorF,'b',  label='Error')
plt.plot(t,line,'r',t,max_plot,'r',t,min_plot,'r',linewidth=2.0, label='Error')
#plt.xlabel('xlabel', fontsize=18)
#plt.ylabel('ylabel', fontsize=16)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('errorF.png')