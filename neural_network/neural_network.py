from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import numpy as np

dataset = np.loadtxt('data/data.csv', delimiter=';')
X = dataset[:, 0:15]
Y = dataset[:, 15]

# Feature scaling with StandardScaler
scale_features_std = StandardScaler()
X = scale_features_std.fit_transform(X)

model = Sequential()

model.add(Dense(20, input_dim=15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=200, batch_size=7, verbose=2)
# Prediction
predictions = model.predict(X)
model.save('model.h5')