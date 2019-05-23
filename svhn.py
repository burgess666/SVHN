import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score
import os.path
import subprocess

# loading data
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

# define models
def create_model():

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()


    '''
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
    '''


    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model


def traintest():
    # Loading train and test data
    X_train, X_test, X_val, y_train, y_test, y_val = loading_data()
    # Train : Test : Validation = 65931 : 26032 : 7326
    print('Train data shape: ', X_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Validation data shape: ', X_val.shape)
    print('label_train shape: ', y_train.shape)
    print('label_test shape: ', y_test.shape)
    print('label_validation shape: ', y_val.shape)

    model = create_model()

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('./', 'checkpoints', '{epoch:03d}-{val_loss:.3f}.h5py'),
        verbose=1,
        save_best_only=True)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5, monitor='val_loss')

    
    # Training
    model.fit(X_train,
              y_train,
              epochs=20,
              batch_size=128,
              validation_data=(X_val, y_val),
              callbacks=[early_stopper, checkpointer],
              verbose=1)

    # predict labels for testing set
    y_predict = model.predict(X_test, batch_size=128)
    average_f1 = f1_score(y_test, y_predict, average='weighted')

if __name__ == '__main__':
    traintest()




