import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import scipy
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.metrics import f1_score
import os.path
import subprocess, datetime
import cv2


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
    '''
    # VGG16 style
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    #model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    #model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    #model.add(layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    '''
    # Alex-style
    # C-BN-P-C-BN-P-C-BN-C-BN-P-C-BN-C-BN-P-FC-FC
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization()) 
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization()) 
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())    
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # Check model details
    model.summary()
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss = 'sparse_categorical_crossentropy',
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

    # create model
    model = create_model()

    # Callback: Save the model.
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('./', 'checkpoints', '{epoch:03d}-vgg-{val_loss:.3f}.h5'),
        verbose=1,
        save_best_only=True)

    # Callback: Stop when we stop learning.
    early_stopper = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')

    # Callback: TensorBoard
    log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Training
    model.fit(X_train,
              y_train,
              epochs=30,
              batch_size=128,
              validation_data=(X_val, y_val),
              callbacks=[early_stopper, checkpointer, tensorboard_callback],
              verbose=1)

    # predict labels for testing set
    y_predict = model.predict_classes(X_test, batch_size=128)
    # average F1 scores for each class
    average_f1 = f1_score(y_test, y_predict, average='weighted')
    print("f1_score:", average_f1)

# Predict a single image
def test(image):
    # Load model
    model_path = 'model_best_alex.h5'
    saved_model = tf.keras.models.load_model(model_path)

    # read image
    img = cv2.imread(image,3)    
    img = cv2.resize(img,(32,32))
    img = np.reshape(img,[1,32,32,3])

    output = saved_model.predict_classes(img)
    return output


if __name__ == '__main__':
    traintest()



