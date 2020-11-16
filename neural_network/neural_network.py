from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np

dataset = np.loadtxt('data/data.csv', delimiter=';')
X = dataset[:, 0:21]
Y = dataset[:, 21]

# Feature scaling with StandardScaler
scale_features_std = StandardScaler()
X = scale_features_std.fit_transform(X)

model = Sequential()

model.add(Dense(30, input_dim=21, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100, batch_size=200, verbose=2)
# Prediction
predictions = model.predict(X)
model.save('model.h5')