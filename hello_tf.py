import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


NB_CLASSES = 10
RESHAPED = 784

model = Sequential()
model.add(Dense(
    NB_CLASSES, input_shape=(RESHAPED,),
    kernel_initializer='zeros',
    name='dense_layer', activation='softmax'))
