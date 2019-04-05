#!/usr/bin/env python
# coding: utf-8

# In[41]:
import numpy as np
import os

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras import optimizers
from keras.layers import Convolution2D, Lambda, Dense, Flatten, Dropout
from keras.models import load_model, Sequential
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt


def model(load, saved_model, shape=(66,200,3)):
    
    if load and saved_model: return load_model(saved_model)
    

    model = Sequential()
    #input normalization layer
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=shape))
    # 3 @ 66x200
    model.add(Convolution2D(filters=24, kernel_size=(5,5), strides=(2,2), 
                    data_format="channels_last", activation="elu"))
    # 24 @ 31x98
    model.add(Convolution2D(filters=36, kernel_size=(5,5), strides=(2,2), 
                    data_format="channels_last", activation="elu"))
    # 36 @ 14x47
    model.add(Convolution2D(filters=48, kernel_size=(5,5), strides=(2,2), 
                    data_format="channels_last", activation="elu"))
    # 48 @ 5x22
    model.add(Convolution2D(filters=64, kernel_size=(3,3), 
                    data_format="channels_last", activation="elu"))
    # 64 @ 3x20
    model.add(Convolution2D(filters=64, kernel_size=(3,3), 
                    data_format="channels_last", activation="elu"))
    # 64 @ 1x18
    model.add(Flatten()) # 1164 neurons

    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="linear"))

    model.add(Dense(1, activation="tanh"))
    
    optim = optimizers.Adam()
    model.compile(loss="mse", optimizer=optim)

    return model

def flip_axis(img,axis):
    if axis == 1:
        new = np.zeros(img.shape)
        dim = img.shape[1]-1
        for i in np.arange(dim):
            new[:,i,:] = img[:,dim-i,:]
    return new

def crop(img):
    shape = img.shape
    for i in np.arange(int(shape[0]/3)):
        img[i,:,:] = 0
    return img

# load image into known format.
def image_handling(path, steering_angle, flip, shape, crop_bool=False):
    """ Image handling """
    image = load_img(path, target_size=shape)
        
    img = img_to_array(image)#/float(255) normalizing with lambda layer instead
    if flip: 
        img = flip_axis(img, 1)
        steering_angle = -steering_angle
    if crop_bool:
        img = crop(img)
    return img, steering_angle
    
def split_data(nr_pts):
    idx = np.arange(nr_pts)
    np.random.shuffle(idx)
    
    tr = int(np.floor(nr_pts*0.8))
    train    = idx[:tr]
    validate = idx[tr:]
    return train, validate

def sample_idx(batch_size, y, proportion):
    i    = 0
    idx  = [0]*batch_size
    flip = [0]*batch_size
    data_num = len(y)
    while i < batch_size:
        candidate = np.random.randint(0,data_num,1)[0]
        #Image of driving forward
        if y[candidate] == 0 and np.random.ranf(1) < (1-proportion)*0.5:
            idx[i] = candidate
            i+=1
        #Image of turning
        elif y[candidate] != 0 and np.random.ranf(1) < proportion:
            idx[i]  = candidate
            flip[i] = np.random.binomial(1,0.5)
            i+=1
    return idx, flip

#Change if augmentation is performed
def _generator(batch_size, X, y, shape, path, proportion):
    while True:
        batch_x   = []
        batch_y   = []
        idx, flip = sample_idx(batch_size, y, proportion) 
        for i, flip_bool in zip(idx,flip):
            x, angle = image_handling(path + os.sep + X[i], y[i], flip_bool, shape=shape)
            batch_x.append(x)
            batch_y.append(angle)
        #print("Left: ",np.sum(np.less(batch_y,0)))
        #print("Forward: ",np.sum(np.equal(batch_y,0)))
        #print("Right: ",np.sum(np.greater(batch_y,0)))
        yield np.array(batch_x), np.array(batch_y)
              
            
def train(path,log):
    shape = (66,200,3)
    front, left, right = np.loadtxt(log, delimiter=",", usecols=[0,1,2], dtype="str", unpack=True)
    angle, forward, backward, speed = np.loadtxt(log, delimiter=",", usecols=[3,4,5,6], unpack=True)

    proportion = np.sum(angle == 0)/float(len(angle))
    
    train, validate = split_data(len(front))
    net = model(load=False, saved_model=None, shape=shape)
    X, y = front[train], angle[train]
    
    X_val, y_val = front[validate], angle[validate]
    
    net.fit_generator(generator        = _generator(256, X, y, shape, path, proportion),
                      validation_data  = _generator(20, X_val, y_val, shape, path, proportion),
                      validation_steps = 20, 
                      epochs = 2, steps_per_epoch=50)
    test_idx, _ = sample_idx(30, y, proportion) 
    for i in test_idx:
        img, _ = image_handling(path + os.sep + X[i], 0, 0, shape)
        img = np.reshape(img, (1,) + shape)
        pred = net.predict(img)
        print("Pred: ", pred, " True: ", y[i])
    net.save('testmodel3.h5')
    


if __name__ == "__main__":
    path = os.getcwd().split(os.sep)[:-1]
    log = path + ["driving_log.csv"]
    img = path + ["IMG"]
    train((os.sep).join(img), (os.sep).join(log))
    #front, left, right = np.loadtxt((os.sep).join(log), delimiter=",", usecols=[0,1,2], dtype="str", unpack=True)
    #shape = (75,320,3)
    #img, _ = image_handling((os.sep).join(img) + os.sep + front[0], 1,0, shape=shape)
    #fig = plt.figure()
    #plt.imshow(img)
    #fig.savefig("test.jpg")


