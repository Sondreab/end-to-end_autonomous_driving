"""
    inspired by:
    https://blog.coast.ai/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a
"""


import csv, random, numpy as np
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt


def get_data_from_csv(data_file):
    #shouldFlip indicates if 50% of the images should be flipped to remove bias
    row_count = sum(1 for row in csv.reader( open(data_file)))
    X = np.empty([row_count,3], dtype=object)
    Y = np.empty([row_count,2])
    with open(data_file) as fin:
        reader = csv.reader(fin)
        counter = 0
        for center_img, left_img, right_img, steering_angle, _, _, speed in reader:
            #we only care about file name, not path
            center_img = center_img.split("/")[-1]
            left_img = left_img.split("/")[-1]
            right_img = right_img.split("/")[-1]
            X[counter] = [center_img, left_img, right_img]
            Y[counter] = [steering_angle, speed]
            counter += 1
    return X, Y


def get_images(X):
    test = X[3][1]
    path = "../data/IMG/" + test
    img = load_img(path, target_size=(160,160))
    #print(img)
    img_print = img_to_array(img)
    print(img_print.shape)
    print(type(img_print))
    plt.imshow(img_print/255.0)
    plt.show()




if __name__ == '__main__':
    #testing purposes
    file_test = "../data/CSV/driving_log.csv"
    X, Y = get_data_from_csv(file_test)

    """
    print(X.shape)
    print(Y.shape)
    print("-----------")
    print(X[3])
    print(Y[3])
    """
    get_images(X)