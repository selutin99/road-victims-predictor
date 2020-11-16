import random

import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from keras.engine.saving import load_model

# Рандомная инициализация
seed = 9
np.random.seed(seed)

# Local path: 'model.h5'
model = load_model('model.h5')

graph = tf.get_default_graph()


def predict(array):
    prediction = model.predict(array)
    print(prediction)

    return prediction


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze')
def analyze():
    return render_template('analyze.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            result = request.form
            array = np.array([[2, float(result['cci']),
                               float(result['cli']), float(result['bci']),
                               float(result['emp_rates']), float(result['inv_to_sales']),
                               float(result['pmi']), float(result['sp500']),
                               float(result['mfg_ord_dur']), float(result['bldg_perm']),
                               float(result['sales']), float(result['fed_funds']),
                               float(result['dow_jones']), float(result['payroll']),
                               float(result['pce'])]])
            predict_result = predict(array)
            predict_result = predict_result[0].item() + random.random()
            predict_result *= 100
            return render_template("result.html", result=predict_result)


app.run()