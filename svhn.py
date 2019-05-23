import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import keras
import keras.backend as K
import time
import os.path
import subprocess

# Define F1 score
def f1_score(y_true, y_pred):

    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)



def loading_data():

    # First, download dataset (train, test and extra)
    if (os.path.exists(os.path.join('./data', 'train_32x32.mat'))):
        print("train_32x32.mat exists")
    else:
        subprocess.run(['wget', '-P', 'data/', 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'])
        
    if (os.path.exists(os.path.join('./data', 'test_32x32.mat'))):
        print("test_32x32.mat exists")
    else:
        subprocess.run(['wget', '-P', 'data/', 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'])
        
    if (os.path.exists(os.path.join('./data', 'extra_32x32.mat'))):
        print("extra_32x32.mat exists")
    else:
        subprocess.run(['wget', '-P', 'data/', 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'])
        

    # Loading data from mat file
    X_train = loadmat(os.path.join('./data', 'train_32x32.mat'))["X"]
    y_train = loadmat(os.path.join('./data', 'train_32x32.mat'))["y"]

    X_test = loadmat(os.path.join('./data', 'test_32x32.mat'))["X"]
    y_test = loadmat(os.path.join('./data', 'test_32x32.mat'))["y"]

    # Normalization
    X_train, X_test = X_train / 255.0, X_test / 255.0
    # Relabel 10 to 0
    y_train[y_train==10] = 0
    y_test[y_test==10] = 0

    # Reshape arrays
    X_train = X_train.transpose((3, 0, 1, 2))
    X_test = X_test.transpose((3, 0, 1, 2))

    # Split origin train set into train set and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
    
    return X_train, X_test, X_val, y_train, y_test, y_val

'''
# View images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
'''


'''model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32,32,3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(10))

model.add(Activation('softmax'))

model.summary()
'''
if __name__ == '__main__':
    
    # Loading train and test data
    X_train, X_test, X_val, y_train, y_test, y_val = loading_data()
    # Train : Test : Validation = 65931 : 26032 : 7326
    print('Train data shape: ', X_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Validation data shape: ', X_val.shape)
    print('label_train shape: ', y_train.shape)
    print('label_test shape: ', y_test.shape)
    print('label_validation shape: ', y_val.shape)
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))


    model.summary()

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('./', 'checkpoints', '{epoch:03d}-{val_loss:.3f}.h5py'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('./', 'logs'))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5, monitor='val_loss')

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('./', 'logs' ,'training-' + str(timestamp) + '.log'))


    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    model.fit(X_train,
              y_train,
              epochs=20,
              batch_size=128,
              validation_data=(X_val, y_val),
              callbacks=[tb, early_stopper, csv_logger, checkpointer],
              verbose=1)
    # Evaluate
    results = model.evaluate(X_test, y_test)
    print(results)



