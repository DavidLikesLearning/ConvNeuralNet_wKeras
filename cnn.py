from keras.datasets import mnist
(train_X,train_Y), (test_X,test_Y) = mnist.load_data()
import numpy as np
from keras.utils import to_categorical
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#this is my first project using keras
#it is predominantly based on the tutorial at datacamp.com from Aditya Sharma made on Dec 5 2017
#https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python 

#Main modifications:
"""
used different dataset (mnist handwritten numbers--the classic Hello World of NN)
changed the structure of the hidden layers based on deeplearning.ai coursework
changed hyperparameters based on coursework

"""

#keras is the library for ml in tensorflow
#matplot for plotting
#numpy for math
classes =np.unique(train_Y)
nClasses =len(classes) # the classes are the 10 arabic numerals

train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
train_X.shape, test_X.shape

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)


train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

epochs = 2 #times to try min loss function
#we had like 10 epochs before and the improvement was 1%
num_classes = 10

#conv2d detects patterns

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.097)) #restricted linear unit, y = 0 then y=x as x surpasses x=0
model.add(MaxPooling2D((4, 2),padding='same')) #2D plane, grabs groups of 4 pixels and then detect more patterns
model.add(Conv2D(64, (2, 2), activation='linear',padding='same')) #another conv for patterns
model.add(LeakyReLU(alpha=0.102)) # ya know
model.add(MaxPooling2D(pool_size=(3, 3),padding='same')) # noise cuz max
model.add(Flatten()) #take 2 by 2 matrix and into vector
model.add(Dense(128, activation='linear'))  #layered neural nets, 128 neurons
model.add(LeakyReLU(alpha=0.215))                  
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()

train = model.fit(train_X, train_label, batch_size=64,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test_loss:', test_eval[0])
print('Test_accuracy:', test_eval[1])

accuracy = train.history['acc'] 
val_accuracy = train.history['val_acc']
loss = train.history['loss']
val_loss = train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Train_accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Valid_accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Train_loss')
plt.plot(epochs, val_loss, 'b', label='Valid_loss')
plt.title('Loss')
plt.legend()
plt.show()

#woohoo 98% accuracy 