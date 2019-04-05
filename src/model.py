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
from keras import models
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Dropout, Convolution2D, Lambda
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


def visualization_model(model, img_tensor):
    
    path = os.getcwd().split(os.sep)[:-1]
    path = path + ['docs'] + ['plots']
    path = (os.sep).join(path)



    os.makedirs(path, exist_ok=True) #location of the plots
    layer_outputs = [layer.output for layer in model.layers[:]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)

    #Special case for the first image
    fig = plt.figure()
    plt.imshow(activations[0][0, :, :, :]) 
    fig.savefig(path + os.sep + "Cropped_input_img")

    layer_names = []
    #Visualizing the conv layers:
    for layer in model.layers[1:9]:
        layer_names.append(layer.name)

    #change these to change the number of channels to visualize
    col_size=5
    row_size=5

    
    for layer_name, layer_activation in zip(layer_names, activations[1:]):
        print(layer_activation.shape)
        print(layer_name)

        act_idx = 0
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
        fig.suptitle(layer_name)
        for col in range(0,col_size):
            for row in range(0,row_size):
                if (act_idx >= layer_activation.shape[3]):
                    break
                ax[row][col].imshow(layer_activation[0, :, :, act_idx], cmap='viridis')
                act_idx +=1
        
        
        #plt.show()
        fig.savefig(path + os.sep + layer_name)



def flip_axis(img,axis):
    if axis == 1:
        new = np.zeros(img.shape)
        dim = img.shape[1]-1
        for i in np.arange(dim):
            new[:,i,:] = img[:,dim-i,:]
    return new


# load image into known format.

def image_handling(path, steering_angle, shape, flip=True):
    """ Image handling """
    image = load_img(path, target_size=shape)
        
    img = img_to_array(image) #/255.0 artefact, this is done by lambda layer

    if flip: 
        img = flip_axis(img, 1)
        steering_angle = -steering_angle
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
            x, angle = image_handling(path + os.sep + X[i], y[i], shape)
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

    #print("y_len: ", len(y))
    #rint("proportion: ", proportion)
    X_val, y_val = front[validate], angle[validate]
    
    net.fit_generator(generator        = _generator(256, X, y, shape, path, proportion),
                      validation_data  = _generator(20, X_val, y_val, shape, path, proportion),
                      validation_steps = 20, 
                      epochs = 2, steps_per_epoch=50)
    
    
    test_idx, _ = sample_idx(50, y, proportion) 
    for i in test_idx:
        img, _ = image_handling(path + os.sep + X[i], 0, shape)
        img = np.reshape(img, (1,) + shape)
        pred = net.predict(img)
        print("Pred: ", pred, " True: ", y[i])
    net.save('testmodel3.h5')

    img_vis, _ = image_handling(path + os.sep + X[1], 0, shape)
    
    img_vis = np.reshape(img_vis, (1,) + shape)
    
    visualization_model(net, img_vis)
    return net
    


if __name__ == "__main__":
    path = os.getcwd().split(os.sep)[:-1]
    log = path + ["driving_log.csv"]
    img = path + ["IMG"]
    print(log)
    print(path)
    net = train((os.sep).join(img), (os.sep).join(log))
    #front, left, right = np.loadtxt((os.sep).join(log), delimiter=",", usecols=[0,1,2], dtype="str", unpack=True)
    #shape = (75,320,3)
    #img, _ = image_handling((os.sep).join(img) + os.sep + front[0], 1,0, shape=shape)
    #fig = plt.figure()
    #plt.imshow(img)
    #fig.savefig("test.jpg")



    


