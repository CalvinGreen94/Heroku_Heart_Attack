import os
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import pandas as pd
import keras
import sklearn
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense,Dropout
from keras.layers.core import Dropout, Activation
import time
from keras.models import Model
from keras.layers.merge import concatenate
NAME = "Shared_input_layer"

data = pd.read_csv('data/framingham_heart_disease.csv')
data = data.fillna(0)
print('COLLECTED FEATURES INCLUDED IN THE DATASET')
data.all()

X = data.drop(['TenYearCHD'],axis=1)
pd.DataFrame(X[:5])

y = data['TenYearCHD']
# y = labelencoder.fit_transform(y)
pd.DataFrame(y[:5])

X = mini.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.32)
batch_size = 710
dropout = 0.47

visible = Input(shape=(15,))
m1 = Dense(6, activation='relu')(visible)
m1 = Dense(6, activation='relu')(m1)
# m1 = Dropout(dropout)(m1)

m2 = Dense(6, activation='relu')(visible)
m2 = Dense(6, activation='relu')(m2)

m3 = Dense(6, activation='relu')(visible)
m3 = Dense(6, activation='relu')(m3)
m3 = Dropout(dropout)(m3)

merge = concatenate([m1,m2,m3],axis=1)

output = Dense(1, activation='relu')(merge)
model = Model(inputs=visible, outputs=output)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
model.summary()
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
model.fit(X_train,y_train,batch_size=batch_size,epochs=100,validation_split=0.43,callbacks=[tensorboard])

model.save('heart_disease_model.h5')
