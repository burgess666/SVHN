import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import scipy
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score
import os.path
import subprocess

# loading data
def loading_data():

    # First, download dataset (train, test and extra)
    if (os.path.exists('train_32x32.mat')):
        print("train_32x32.mat exists")
    else:
        subprocess.run(['wget', '-P', 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'])
        
    if (os.path.exists('test_32x32.mat')):
        print("test_32x32.mat exists")
    else:
        subprocess.run(['wget', '-P', 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'])
        
    # Loading data from mat file
    X_train = loadmat('train_32x32.mat')["X"]
    y_train = loadmat('train_32x32.mat')["y"]

    X_test = loadmat('test_32x32.mat')["X"]
    y_test = loadmat('test_32x32.mat')["y"]

    # Normalization
    X_train, X_test, X_extra = X_train / 255.0, X_test / 255.0
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
    '''
    # VGG16 style
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    '''
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization()) 
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization()) 
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Take a look at the model summary
    model.summary()
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model


def traintest():
    # Loading train and test data
    X_train, X_test, X_val, y_train, y_test, y_val = loading_data()
    # Train : Test : Validation = 65931 : 26032 : 7326
    '''
    Train data shape:  (65931, 32, 32, 3)
    Test data shape:  (26032, 32, 32, 3)
    Validation data shape:  (7326, 32, 32, 3)
    label_train shape:  (65931, 1)
    label_test shape:  (26032, 1)
    label_validation shape:  (7326, 1)
    '''
    print('Train data shape: ', X_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Validation data shape: ', X_val.shape)
    print('label_train shape: ', y_train.shape)
    print('label_test shape: ', y_test.shape)
    print('label_validation shape: ', y_val.shape)

    model = create_model()

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('./', 'checkpoints', '{epoch:03d}-3-{val_loss:.3f}.h5py'),
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
    #y_predict = model.predict(X_test, batch_size=128)
    score = model.evaluate(X_test, y_test, verbose=0)
    #average_f1 = f1_score(np.transpose(y_test), np.transpose(y_predict), average='weighted')
    print ("score:")
    print (score)


def test(image):
    # Load model
    model_path = './'
    saved_model = tf.keras.models.load_model(model_path)

    # read image
    read_image = scipy.misc.imread(image)
    output = saved_model.predict(read_image)
    return output


if __name__ == '__main__':
    traintest()





