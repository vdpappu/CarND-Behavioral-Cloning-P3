import numpy as np
import pandas as pd
import cv2
import itertools
import csv
from glob import glob
import os,sys

print('Running this from: ', os.getcwd())


path = '/home/tebd/Documents/sdc/CarND-Behavioral-Cloning-P3/BC_Training_Data/track2//'
subdirectories = os.listdir(path)
master_list = list()
master_df = pd.DataFrame()

for dir_ in subdirectories:
    t_df = pd.read_csv(path+dir_+"/driving_log.csv",header=None)
    #del t_df['index']
    print(len(t_df)*3)
    t_df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    master_list.append(t_df)
    
master_df = pd.concat(master_list)
master_df = master_df.reset_index()

assert len(master_df) == len(master_list[0])+len(master_list[1])

del master_df['index']

master_df.to_csv('./data/driving_log_add_t2_d.csv',index=False)
