import numpy as np
import pandas as pd
import os
import cv2
import sklearn
import tensorflow as tf
import itertools
import csv
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.model_selection import train_test_split
path = os.getcwd()
os.getcwd()


threshold = 0.15 #Angle threshold, adjust here and generate data
in_data_1 = pd.read_csv('./data/driving_log.csv')
in_data_2 = pd.read_csv('./data/driving_log_add_t2_d.csv') #Track2 proper driving

in_data = pd.concat([in_data_1,in_data_2])
in_data_strait = in_data[abs(in_data['steering'])<=threshold]
in_data_recovery = in_data[abs(in_data['steering'])>threshold]


# In[7]:

#Adjust right and left camera images with correction of 0.2

imgs_strait = list(pd.concat([in_data_strait['center'],in_data_strait['left'],in_data_strait['right']]))
steer_angle_strait = list(pd.concat([in_data_strait['steering'],in_data_strait['steering']+0.2,in_data_strait['steering']-0.20]))

imgs_recovery = list(pd.concat([in_data_recovery['center'],in_data_recovery['left'],in_data_recovery['right']]))
steer_angle_recovery = list(pd.concat([in_data_recovery['steering'],in_data_recovery['steering']+0.2,in_data_recovery['steering']-0.2]))


# In[8]:

df_strait = pd.DataFrame(steer_angle_strait,imgs_strait)
df_strait = df_strait.reset_index()
df_strait.columns = ['Image','steering']
df_strait.head()
df_strait = df_strait.iloc[np.random.permutation(len(df_strait))]
df_strait.to_csv("./data/img_steer_strait.csv",index=False,header=False)


df_recovery = pd.DataFrame(steer_angle_recovery,imgs_recovery)
df_recovery = df_recovery.reset_index()
df_recovery.columns = ['Image','steering']
df_recovery.head()
df_recovery = df_recovery.iloc[np.random.permutation(len(df_recovery))]
df_recovery.to_csv("./data/img_steer_recovery.csv",index=False,header=False)

#idx = int(0.15*len(df_strait))
#df_strait = df_strait[0:idx]

#df_merged = pd.concat([df_strait,df_strait])
#df_merged.to_csv("./data/img_steer_new.csv",index=False,header=False)


# In[ ]:

len(df_recovery)


# In[ ]:

len(df_strait)


# In[ ]:



