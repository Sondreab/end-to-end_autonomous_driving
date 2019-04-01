#!/usr/bin/env python
# coding: utf-8

# In[41]:


from keras.layers import Dense, Flatten
from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

def model(load, saved_model, shape):
    
    if load and saved_model: return load_model(saved_model)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=shape))
    model.add(MaxPooling2D())
    #model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D())
    #model.add(BatchNormalization(axis=1))
    model.add(Conv2D(128,(3, 3), activation="relu"))
    model.add(MaxPooling2D())
    #model.add(BatchNormalization(axis=1))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
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
def _generator(batch_size, X, y, shape):
    while True:
        batch_x = []
        idx     = np.random.randint(0,len(X), max([batch_size,len(X)]))
        batch_y = y[idx]
        for i in idx:
            batch_x.append(image_handling(X[i], shape=shape))
        
        yield np.array(batch_x), np.array(batch_y)
              
def train():
    shape = (40,60,3)
    front, left, right = np.loadtxt("C:/Users/fschj/OneDrive/Skrivebord/term1-simulator-windows/driving_log.csv",
                                        delimiter=",", usecols=[0,1,2], dtype="str", unpack=True)
    angle, forward, backward, speed = np.loadtxt("C:/Users/fschj/OneDrive/Skrivebord/term1-simulator-windows/driving_log.csv",
                                                delimiter=",", usecols=[3,4,5,6], unpack=True)

    train, validate, test = split_data(len(front))
    net  = model(load=False, saved_model=None, shape=shape)
    X, y = front[train], np.divide(angle[train],50) + 0.5
    
    X_val, y_val = front[validate], np.divide(angle[validate],50) + 0.5
    
    net.fit_generator(generator        = _generator(64, X, y, shape),
                      validation_data  = _generator(len(validate),X_val, y_val, shape),
                      validation_steps = 10, 
                      epochs = 5, steps_per_epoch=len(front))
    net.save('save/testmodel.h5')
    

# split into sets of data 

# data augmentation / processing? --

#set up network v

#make train module

if __name__ == "__main__":
    train()

#train and test module if converged

# compile model and load into udacity


# In[8]:





# In[ ]:




