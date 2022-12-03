#Prediction model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import ecdata


x_train = ecdata.indata()    #input 풍속
y_train = ecdata.outdata()  #output U

print(np.shape(x_train))
print(np.shape(y_train))


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(8,activation="linear")    
    ])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(x_train,y_train, epochs=800)

model.summary()

pred = model.predict(np.array([[-1.7,1.9]]))
print(pred)

# predict = ae.decoder(pred).numpy()
# np.savetxt('vtodecoder.txt',predict,delimiter=" ")