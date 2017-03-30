import pandas as pd
from numpy import argmax

from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


train_data = pd.read_csv('./tri_data.csv', delimiter=';')

x = train_data[train_data.keys()[:-1]].values
y = train_data['result'].values


encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = Sequential([
    Dense(64, input_dim=3),
    Activation('relu'),
    Dense(4),
    Activation('softmax')
])

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy'
)

model.fit(x_train, y_train)

pred = model.predict(x_test)

pred = argmax(pred, axis=1)

pred = encoder.inverse_transform(pred)