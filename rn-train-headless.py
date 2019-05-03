import numpy as np
import keras
from tensorflow.python.client import device_lib
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
from keras.layers import  Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
get_available_gpus()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))



def lr_decay(epoch):
	initial_lrate = 0.1
	drop = 0.75
	epochs_drop = 10.0
  	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate


base_model = ResNet50(weights='imagenet',include_top=False,
              classes=10,input_shape=(32, 32, 3))
convf=base_model.output
poolf=GlobalAveragePooling2D()(convf)
densef=Dense(1000,activation='relu')(poolf) #we add dense layers so that the model can learn more complex functions and classify for better results.
outputf=Dense(10,activation='softmax')(densef) #final layer with softmax activationm

model=Model(inputs=base_model.input,outputs=outputf)
model.summary()

sgd = SGD(lr=lr_decay(0), decay=0, momentum=0.9, nesterov=True)
history=LossHistory()

callback_list  = [LearningRateScheduler(lr_decay),LossHistory()]

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'],
	      callbacks = callback_list)
              

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255
x_test = x_test/255

batch_size = 64
num_classes = 10
epochs = 50
print(x_train[0:1].shape)
print(model.predict(x_train[0:1]))
print(y_train[0:1])
model.fit(x_train, y_train,
        batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
from keras.models import load_model
print(y_train[0:1])

print(model.predict(x_train[0:1]))
model.save('hot_model_sgd.h5') 
