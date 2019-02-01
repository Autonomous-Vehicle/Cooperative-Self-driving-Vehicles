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
df = pd.read_csv("error.csv", dtype={'error1': np.float, 'error2': np.float, 'error3': np.float, 'error4': np.float})
#df = df[df['frame_id'] == 'center_camera'].reset_index(drop=True)

#fnames = np.array('Ch2/' + df['filename'])
error1 = np.array(df['error1'])
#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)
len_x=len(error1)
plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([0, 1.5])
plt.xlim([0, len_x])
plt.plot(error1, label='Error')
#plt.xlabel('xlabel', fontsize=18)
#plt.ylabel('ylabel', fontsize=16)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('error1.png')


error2 = np.array(df['error2'])
#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)

plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([0, 1.5])
plt.xlim([0, len_x])
#plt.xticks([0.4,0.14,0.2,0.2], fontsize = 50)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
plt.plot(error2, label='Error')
#plt.ylabel('ylabel', fontsize=16)
#plt.legend()
plt.savefig ('error2.png')

error3 = np.array(df['error3'])
#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)
maximun=max(error3)
plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([0, maximun])
plt.xlim([0, len_x])
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
plt.plot(error3, label='Error')
#plt.legend()
plt.savefig ('error3.png')


error4 = np.array(df['error4'])
#for ix, tr in enumerate(zip(error1)):
#    error= val_output(tr)
#    real.append(error)

plt.figure(figsize=(16, 9))
#plt.xlim([0, 33000])
plt.ylim([0, 1.5])
plt.xlim([0, len_x])
plt.plot(error4, label='Error')
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
#plt.legend()
plt.savefig ('error4.png')

print("Mean Error: ", np.sqrt(np.mean((error1) ** 2)))
print("MAE: ", (np.mean(abs(error1) )))

