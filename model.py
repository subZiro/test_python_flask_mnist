import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds  # pip3 install tensorflow-datasets данные для обучения
import tensorflow as tf
import logging
import numpy as np

#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.get_logger().setLevel(logging.ERROR)





'''#построение модели
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
#print(model.summary())

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])
model.fit(X, y, batch_size=1, epochs=1000, verbose=0)


print("Network test:")
print("XOR(0,0):", model.predict_proba(np.array([[0, 0]])))
print("XOR(0,1):", model.predict_proba(np.array([[0, 1]])))
'''




'''
# первая версия
def mnist_make_model(image_w: int, image_h: int):
   # Neural network model
   model = Sequential()
   model.add(Dense(784, activation='relu', input_shape=(image_w*image_h,)))
   model.add(Dense(10, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
   return model


def mnist_mlp_train(model):
   (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
   # x_train: 60000x28x28 array, x_test: 10000x28x28 array
   image_size = x_train.shape[1]
   train_data = x_train.reshape(x_train.shape[0], image_size*image_size)
   test_data = x_test.reshape(x_test.shape[0], image_size*image_size)
   train_data = train_data.astype('float32')
   test_data = test_data.astype('float32')
   train_data /= 255.0
   test_data /= 255.0
   # encode the labels - we have 10 output classes
   # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
   num_classes = 10
   train_labels_cat = keras.utils.to_categorical(y_train, num_classes)
   test_labels_cat = keras.utils.to_categorical(y_test, num_classes)
   print("Training the network...")
   t_start = time.time()

   # Start training the network
   model.fit(train_data,
             train_labels_cat,
             epochs=8,
             batch_size=64,
             verbose=1,
             validation_data=(test_data, test_labels_cat))

model = mnist_make_model(image_w=28, image_h=28)
mnist_mlp_train(model)
model.save('data/mlp_digits_28x28.h5')
'''


'''
проверка загрузки тренировочных данных
path = '/data/mnist.npz'
path = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
print(len(train_images))
'''







def mnist_cnn_model():
   image_size = 28
   num_channels = 1  # 1 for grayscale images
   num_classes = 10  # Number of outputs
   model = Sequential()
   model.add(Conv2D(filters=32,
                    kernel_size=(3,3),
                    activation='relu',
                    padding='same',
                    input_shape=(image_size, image_size, num_channels)))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64,
                    kernel_size=(3, 3),
                    activation='relu',
                    padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Flatten())
   # Densely connected layers
   model.add(Dense(128, activation='relu'))
   # Output layer
   model.add(Dense(num_classes, activation='softmax'))
   model.compile(optimizer=Adam(), loss='categorical_crossentropy',
            metrics=['accuracy'])
   return model


def mnist_cnn_train(model):
   (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()
   # Get image size
   image_size = 28
   num_channels = 1  # 1 for grayscale images
   # re-shape and re-scale the images data
   train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
   train_data = train_data.astype('float32') / 255.0
   # encode the labels - we have 10 output classes
   # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
   num_classes = 10
   train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)
   # re-shape and re-scale the images validation data
   val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
   val_data = val_data.astype('float32') / 255.0
   # encode the labels - we have 10 output classes
   val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)
   print("Training the network...")
   t_start = time.time()
   # Start training the network
   model.fit(train_data,
             train_labels_cat,
             epochs=8,
             batch_size=64,
             validation_data=(val_data, val_labels_cat))
   print("Done, dT:", time.time() - t_start)
   return model



def cnn_digits_predict(model, image_file):
    """предсказание по файлу"""
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file,
                                             target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))
    result = model.predict_classes([img_arr])
    return result

def model_to_predict():
    model = tf.keras.models.load_model('data/cnn_digits_28x28.h5')
    return model


if __name__ == '__main__':
    # создание модели и сохранение в файл
    model = mnist_cnn_model()
    mnist_cnn_train(model)
    model.save('data/cnn_digits_28x28.h5')
    print('Complited')