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
            array = np.array([[0, float(result['typ_cod']),
                               float(result['nlevel']), float(result['btf']),
                               float(result['oneway']), float(result['surface']),
                               float(result['splitter']), float(result['speedlim']),
                               float(result['f_rspeed']), float(result['t_rspeed']),
                               float(result['lanewidth']), float(result['f_lanes']),
                               float(result['t_lanes']), float(result['f_sidewalk']),
                               float(result['t_sidewalk']), float(result['f_buslanes']),
                               float(result['t_buslanes']), float(result['multidigit']),
                               float(result['f_parking']), float(result['t_parking']),
                               float(result['bicyclanes'])]])
            predict_result = predict(array)
            predict_result = predict_result[0].item()
            return render_template("result.html", result=predict_result)


app.run()
