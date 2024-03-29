"""
Author :A cup of tea
file:training a cnn articature using cifer-10 dataset

"""

import keras
import numpy as np
#project modules
from .. import config
from . import my_model, preprocess

# loading data
x_train, y_train = preprocess.load_tain_data()
print("train data shape: ", x_train.shape)
print("train data label: ", y_train.shape)
# loading model
model = my_model.get_model()
# compile
model.compile(keras.optimizers.Adam(config.lr),
              keras.losses.categorical_crossentropy,
               metrics=["accuracy"])

#checkpoints
model_cp = my_model.save_model_checkpoint()
early_stopping = my_model.set_early_stopping()
# for training model
model.fit(x_train, y_train,
        batch_size=config.batch_size, 
        epochs=config.nb_epochs, 
        verbose=2, 
        shuffle = True,
        callbacks=[early_stopping,model_cp], 
        validation_split=0.2)

        