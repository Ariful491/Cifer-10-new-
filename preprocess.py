import numpy as np

import cv2, os
from cv2 import cv2

import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# project package 
from .. import config

x_train = np.ndarray((config.nb_train_samples,
                config.img_size,config.img_size,
                config.img_channel),dtype=np.float32)

def normalization(x):
    x = np.divide(x,255.0)
    x = np.subtract(x,0.5)
    x = np.multiply(x,2.0)
    return x

# loading train data...
def load_tain_data():
    train_data_dir =os.path.join(config.dataset_path(),"train")
    # print(os.listdir(train_data_dir))

    train_images = sorted(os.listdir(train_data_dir),
            key = lambda x:int(x.split(".")[0]))
    # print(train_images)
    train_images = [os.path.join(train_data_dir,img_path) 
                    for img_path in train_images]



    # print(train_images)
    # loading image labels from csv file........................
    # ......................__..........
    train_labels_df = pd.read_csv(os.path.join(config.dataset_path(),
                              "trainLabels.csv"))

    train_labels = train_labels_df["label"].values



    # on hot encoding........................
    # ....................
    encoder = LabelBinarizer()
    y_labels =encoder.fit_transform(train_labels)


    print("Arif vaiyar train images loading colteche please wait.........")

# loading images from absolute directory.................
# ..................using opern Cv library .................../////////////


    for i, img_dir in enumerate(train_images):
        img = cv2.imread(img_dir)
        img = normalization(img)

       
        x_train[i] = img 

    return x_train, y_labels

# loading test data............
def load_test_data():
    test_data_dir =os.path.join(config.dataset_path(),"test")
    # print(os.listdir(test_data_dir))

    test_images = sorted(os.listdir(test_data_dir),
            key = lambda x:int(x.split(".")[0]))
    # print(test_images)
    test_images = [os.path.join(test_data_dir,img_path) 
                    for img_path in test_images]



    start = 0
    nb_images = 50000
    for part in range(0, 6):

        if not (part== 0):start+=nb_images
        end =start + nb_images

    
        x_test=np.ndarray((nb_images,
                        config.img_size,config.img_size,
                        config.img_channel),dtype=np.float32)

        print("loading test image from",start ," to ", end)
    #    loading image from absoulate directory
        for i , img_dir in enumerate(test_images[start:end]):
            img = cv2.imread(img_dir)
            img = normalization(img)
            x_test[i] = img

        np.save(os.path.join(config.output_path(),
                    "x_test_"+ str(part)), x_test)
        del x_test
    

# loading test mage form numpy array 
def get_test_data_by_part(part):
    print("loading test image part",part,"form numpy array")
    return np.load(os.path.join(config.output_path(),
                     "x_test_"+ str(part) + ".npy"))

if __name__ == "__main__":
    load_test_data()
    

