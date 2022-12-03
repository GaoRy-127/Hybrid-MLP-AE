# autoEncoder for CFD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, losses
from keras.models import Model
import udata
import os

# train data
x_train = udata.outdata()
x_test = x_train
y_data = udata.indata()

print(np.shape(x_train))

# autoencoder model

latent_dim = 8
input_size = len(x_train[0])
layer_size = input_size * 3


class Autoencoder(Model):
  def __init__(self, encoding_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(latent_dim, activation='tanh'),
    ])
    self.decoder = tf.keras.Sequential([
        layers.Dense(layer_size, activation='linear'),
        layers.Reshape((input_size,3))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train, x_train,
                epochs=1000,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_u = autoencoder.encoder(x_test).numpy()
decoded_u = autoencoder.decoder(encoded_u).numpy()

os.chdir('../')
os.chdir('result')

print(encoded_u)
getdata=np.hstack((y_data,encoded_u))
np.savetxt('invisible.txt',getdata,delimiter=",")
# print(getdata)  # 풍속과 은닉층 값 합친것

print(np.shape(decoded_u))
test1cfd = decoded_u[0,:,:]
np.set_printoptions(formatter={'float_kind': lambda x:"{0:0.5f}".format(x)})
test1cfd = np.round_(test1cfd,5)
# print(test1cfd.dtype)
print(test1cfd)
np.savetxt('test1.txt', test1cfd, delimiter=" ")
# test2cfd = decoded_u([-0.99999803 -0.99877566  0.96006227 -0.9995706  -1.          0.9999979
#    0.96989375 -0.97782224])
print(y_data)
