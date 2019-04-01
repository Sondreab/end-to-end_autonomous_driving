#!/usr/bin/env python
# coding: utf-8

# In[41]:

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from keras import optimizers
from keras.layers import Dense, Flatten, Dropout
from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

def model(load, saved_model, shape):
    
    if load and saved_model: return load_model(saved_model)
    
    "Dobbelt opp med parametere funker men grisetregt"
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation="relu", input_shape=shape))
    model.add(MaxPooling2D())
    #model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D())
    #model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64,(3, 3), activation="relu"))
    model.add(MaxPooling2D())
    #model.add(BatchNormalization(axis=1))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    optim = optimizers.Adam()
    model.compile(loss="mse", optimizer=optim)
    return model


# load image into known format.
def image_handling(path, steering_angle=0, augment=0, shape=(100,200)):
    """ Image handling """
    image = load_img(path, target_size=shape)
    
    img = img_to_array(image)
    
    return img
    
def split_data(nr_pts):
    idx = np.arange(nr_pts)
    np.random.shuffle(idx)
    
    tr = int(np.floor(nr_pts*0.75))
    va = int(np.floor(nr_pts*0.9))
    train    = idx[:tr]
    validate = idx[tr:va]
    test     = idx[va:]
    
    return train, validate, test

#Change if augmentation is performed
def _generator(batch_size, X, y, shape, path):
    while True:
        batch_x = []
        idx     = np.random.randint(0,len(X), min([batch_size,len(X)]))
        batch_y = y[idx]
        for i in idx:
            batch_x.append(image_handling(path + "/" + X[i], shape=shape))
        
        yield np.array(batch_x), np.array(batch_y)
              
def train(path,log):
    shape = (40,60,3)
    front, left, right = np.loadtxt(log, delimiter=",", usecols=[0,1,2], dtype="str", unpack=True)
    angle, forward, backward, speed = np.loadtxt(log, delimiter=",", usecols=[3,4,5,6], unpack=True)

    train, validate, test = split_data(len(front))
    net  = model(load=False, saved_model=None, shape=shape)
    X, y = front[train], np.divide(angle[train],50) + 0.5
    
    X_val, y_val = front[validate], np.divide(angle[validate],50) + 0.5
    
    net.fit_generator(generator        = _generator(64, X, y, shape, path),
                      validation_data  = _generator(10,X_val, y_val, shape, path),
                      validation_steps = 10, 
                      epochs = 5, steps_per_epoch=int(len(front)/5))
    net.save('save/testmodel.h5')
    


if __name__ == "__main__":
    path = os.getcwd().split("/")[:-1]
    log = path + ["driving_log.csv"]
    train("/".join(path), "/".join(log))



