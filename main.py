import pandas as pd

from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
# keras.utils.np_utils.to_categorical
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('./data.csv', delimiter=';')

X = train_data[train_data.keys()[:-1]].values
y = train_data['결과'].values

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Sequential([
    Dense(32, input_dim=5),
    Activation('relu'),
    Dense(3),
    Activation('softmax')
])

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
)

model.fit(X_train, y_train)

pred = model.predict(X_test)


print(pred)
