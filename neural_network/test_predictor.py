import csv
import random

import numpy as np
from keras.engine.saving import load_model


def predict(array):
    prediction = model.predict(array)
    return prediction


if __name__ == '__main__':
    random.seed(0)
    model = load_model('model.h5')

    test_dataset = np.loadtxt('data/test_data.csv', delimiter=';')

    with open('data/test_prediction.csv', "a", newline="") as file:
        writer = csv.writer(file, delimiter=";")

        for i in test_dataset:
            array = np.array([[0, float(i[2]),
                               float(i[3]), float(i[4]),
                               float(i[5]), float(i[6]),
                               float(i[7]), float(i[8]),
                               float(i[9]), float(i[10]),
                               float(i[11]), float(i[12]),
                               float(i[13]), float(i[14]),
                               float(i[15]), float(i[16]),
                               float(i[17]), float(i[18]),
                               float(i[19]), float(i[20]),
                               float(i[21])]])
            predict_result = predict(array)
            predict_result = predict_result[0].item()
            print('For id {0} prediction is {1}'.format(int(i[0]), int(predict_result)))
            writer.writerow([int(i[0]), int(predict_result)])
