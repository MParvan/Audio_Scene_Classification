

#### Import Libraries ####
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import wavfile
import scipy.io as sio
import glob

#### List Your Data Files ####
List = glob.glob('*.wav')


#### Allocate Train & Test Data ####
f = sio.loadmat("fold3026-matrices_v7.mat")

indiceMa = np.array(f["indiceMa"])
indiceMt = np.array(f["indiceMt"])


#### Load Your Data ####
x = np.zeros((3026, 44100))
i = 0

for item in List:
    fs, data = wavfile.read(item)
    data = data[44100:44100+int(np.size(data)/15)]
    x[i,:] = data
    i = i+1


#### Data Labeling ####
y = np.zeros(np.size(x, 0))
y[23:np.size(x,0)] = 1
y[215:np.size(x,0)] = 2
y[358:np.size(x,0)] = 3
y[478:np.size(x,0)] = 4
y[721:np.size(x,0)] = 5
y[809:np.size(x,0)] = 6
y[1078:np.size(x,0)] = 7
y[1223:np.size(x,0)] = 8
y[1499:np.size(x,0)] = 9
y[1638:np.size(x,0)] = 10
y[1887:np.size(x,0)] = 11
y[2042:np.size(x,0)] = 12
y[2132:np.size(x,0)] = 13
y[2265:np.size(x,0)] = 14
y[2387:np.size(x,0)] = 15
y[2590:np.size(x,0)] = 16
y[2754:np.size(x,0)] = 17
y[2901:np.size(x,0)] = 18


#### Allocate Train & Test Data ####
index_train = indiceMa[1,:]-1
x_train = x[index_train,:]
y_train = y[index_train]

index_test = indiceMt[1,:]-1
x_test = x[index_test,:]
y_test = y[index_test]

del x




#### Denoising ####
import cv2
x_tr = cv2.medianBlur(np.float32(x_train), 5)

x_te = cv2.medianBlur(np.float32(x_test), 5)

del data



#### melspectrogram ####
import librosa
import librosa.display


x_train_mel = np.zeros((2419, 64, 81))
for i in range(np.size(x_train, 0)):
    spect = librosa.feature.melspectrogram(y=x_train[i, :], sr=22050, 
                                       S=None, n_fft=int(0.05*22050), n_mels=64,
                                       hop_length=int(0.05*11025),
                                       power=2.0);
   
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    x_train_mel[i, :, :] = mel_spect

del x_train


x_tr_mel = np.zeros((2419, 64, 81))
for i in range(np.size(x_tr, 0)):
    spect = librosa.feature.melspectrogram(y=x_tr[i, :], sr=22050, 
                                       S=None, n_fft=int(0.05*22050), n_mels=64,
                                       hop_length=int(0.05*11025),
                                       power=2.0);
   
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    x_tr_mel[i, :, :] = mel_spect

del x_tr



x_test_mel = np.zeros((607, 64, 81))
for i in range(np.size(x_test, 0)):
    spect = librosa.feature.melspectrogram(y=x_test[i, :], sr=22050, 
                                       S=None, n_fft=int(0.05*22050), n_mels=64,
                                       hop_length=int(0.05*11025),
                                       power=2.0);
   
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    x_test_mel[i, :, :] = mel_spect

del x_test


x_te_mel = np.zeros((607, 64, 81))
for i in range(np.size(x_te, 0)):
    spect = librosa.feature.melspectrogram(y=x_te[i, :], sr=22050, 
                                       S=None, n_fft=int(0.05*22050), n_mels=64,
                                       hop_length=int(0.05*11025),
                                       power=2.0);
   
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    x_te_mel[i, :, :] = mel_spect

del x_te


#### Preparing Data for Neural Network ####
x_train = np.zeros((2419,2,64,81))
for i in range(2419):
    x_train[i,:,:,:] = (np.concatenate((x_train_mel[i,:,:], x_tr_mel[i,:,:]),axis=0)).reshape((2,64,81))

x_test = np.zeros((607,2,64,81))
for i in range(607):
    x_test[i,:,:,:] = (np.concatenate((x_test_mel[i,:,:], x_te_mel[i,:,:]),axis=0)).reshape((2,64,81))


del x_train_mel
del x_test_mel
del x_tr_mel
del x_te_mel



num_classes = 19
epochs = 3
batch_size = 8
validation_split = 0.3

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.reshape((2419,64,81,2))
x_test = x_test.reshape((607,64,81,2))


#### Import Libraries for Neural Network ####
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.layers import Conv2D, MaxPooling2D, GRU, Bidirectional
from keras.layers import BatchNormalization, Activation, GlobalMaxPooling1D
from keras_self_attention import SeqSelfAttention



model = Sequential()

#layer1
model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(64, 81, 2), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(4, 1)))


## Layer 2
model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(4, 1)))


## Layer 3
model.add(Conv2D(256, kernel_size=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(4, 1)))


model.add(Reshape((model.output_shape[2],model.output_shape[3])))


## Layer 4
model.add(Bidirectional(GRU(128, input_shape=(256, ), activation='tanh', recurrent_activation='sigmoid',
              dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))

## Layer 5
model.add(Bidirectional(GRU(128, input_shape=(64, ), activation='tanh', recurrent_activation='sigmoid',
              dropout=0.1, recurrent_dropout=0.1, return_sequences=True)))



model.add(SeqSelfAttention(attention_activation='tanh'))

model.add(GlobalMaxPooling1D(data_format='channels_last'))



model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.summary()


from keras import losses

model.compile(loss=losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001, decay=0.0),
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=validation_split)



score_test = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])

score_train = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1])

target = model.predict(x_test, batch_size=batch_size)






############  SVM PART  ###############


#### Extracting After-Flatten Data ####
model.layers[19].name = "layer"

from keras.models import Model

layer_name = "layer"
intermediate_layer = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)


intermediate_output_train = intermediate_layer.predict(x_train)
intermediate_output_test = intermediate_layer.predict(x_test)

#### SVM ####
from sklearn import svm
cls = svm.LinearSVC(C = 0.1)

cls.fit(intermediate_output_train, y_train)
out = cls.predict(intermediate_output_test)










