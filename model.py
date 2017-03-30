import numpy as np
import pandas as pd
import os
import cv2
import sklearn
import tensorflow as tf
import itertools
import csv
import random

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Activation, Dropout,ELU
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.convolutional import MaxPooling2D as Mpool2D
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF
from keras.layers.core import Dropout

from sklearn.model_selection import train_test_split

def get_session(gpu_fraction=0.4):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

print("Running this from: ", os.getcwd())
path = os.getcwd()

def adaptive_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

#horizantal and verfical shifts
def trans_image(image,steer,trans_range,cols=80,rows=80):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 20*np.random.uniform()-20/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang


# In[24]:

#Run dataPreparation.ipynb before this to generate strait and recovery csvs

#split the csv into two one with thresholds >  abs(0.15)
lines_strait = []
lines_recovery = []

#read strait and recovery files, merge, split and freakout!!

with open('./data/img_steer_strait.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines_strait.append(line)

with open('./data/img_steer_recovery.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines_recovery.append(line)
        
#Strip away ~80% of strait samples
random.shuffle(lines_strait)
random.shuffle(lines_recovery)
strait_perc = 0.1
lines_strait = lines_strait[0:int(len(lines_strait)*strait_perc)]

samples = lines_strait+lines_recovery
random.shuffle(samples)

train_samples,validation_samples = train_test_split(samples,test_size = 0.25)
print("Lenght_train_samples: ", len(train_samples))
print("Length_Validation_samples: ",len(validation_samples))


# In[16]:

#lines = []

#with open('./data/img_steer_new.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)
        
#random.shuffle(lines)
#train_samples,validation_samples = train_test_split(lines,test_size = 0.2)
#len(train_samples)


# In[ ]:

#for i in range(len(train_samples)):
#    name = train_samples[i][0].split('/')[-1]
#       #name = "data/"+train_samples[0][0]
#    print(name)
#    img = cv2.imread("data/IMG/"+name)
#    img.shape == (160, 320, 3)


# ### Utility function to create a separate dataframe with images and corresponding steering predictions

# In[25]:

def generator(samples, batch_size=64):

    while 1: # Loop forever so the generator never terminates
        #shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0].split('/')[-1]
                center_image = cv2.imread("./data/IMG/"+name)
                #center_image = center_image/255
                #center_image = center_image-0.5
                center_image = center_image[40:140, ]
                center_image = cv2.resize(center_image, (80,80))
                center_angle = float(batch_sample[1])
                prob = np.random.uniform()
                if prob < 0.35:
                    center_image = adaptive_brightness(center_image)
                    #center_image = cv2.flip(center_image,1)
                    #center_angle = -center_angle
                if prob >  0.35 and prob<=0.7:
                    center_image,center_angle = trans_image(center_image,center_angle,30)
                flip_prob = np.random.uniform()
                if flip_prob < 0.30:
                    center_image = cv2.flip(center_image,1)
                    center_angle = -center_angle
                center_image = center_image.reshape(3,80,80)
                #center_image = center_image/255
                #center_image = center_image-0.5
                images.append(center_image)
                angles.append(center_angle)
                #assert center_image.shape == (80,80,3)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[26]:

#X_train,y_train = sklearn.utils.shuffle(X_train,y_train)
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)
#next(validation_generator)


# In[27]:

#Le-Net architecture
#with K.tf.device('/gpu:1'):
#    K._set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))

#steps_per_epoch = (len(train_samples)*4)/64 #batch size
#validation_steps = len(validation_samples)/64

#Try 
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(3,80,80)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Conv2D(48,5,5, border_mode="valid"))
model.add(Dropout(.5))
model.add(ELU())
model.add(Mpool2D((2, 2), border_mode='valid'))

# layer 2 output shape is 15x15x16
model.add(Conv2D(64,3,3, border_mode="valid"))
model.add(Dropout(.5))
model.add(ELU())
model.add(Mpool2D((2, 2), border_mode='valid'))

# layer 3 output shape is 12x12x16
model.add(Conv2D(128,3,3, border_mode="valid"))
model.add(Dropout(.5))
model.add(ELU())
model.add(Mpool2D((2, 2), border_mode='valid'))

#model.add(Conv2D(256,3,3, border_mode="valid"))
#model.add(Dropout(.5))
#model.add(ELU())
#model.add(Mpool2D((2, 2), border_mode='valid'))


# Flatten the output
model.add(Flatten())

# layer 4
model.add(Dense(512))
model.add(ELU())
# layer 5
model.add(Dense(64))
model.add(ELU())
model.add(Dense(16))
model.add(ELU())

model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*2,
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=15)

#model.fit_generator(generator=train_generator,steps_per_epoch=steps_per_epoch,
#                    epochs=10,validation_data=validation_generator,
#                    validation_steps = validation_steps)
model.save('Model_17DownSample.h5')


# In[ ]:

for layer in model.layers:
    print(layer,"--->",layer.output_shape)


# In[ ]:

os.getcwd()


# In[ ]:



