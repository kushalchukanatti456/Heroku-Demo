import numpy as np
import pickle
from flask import Flask, request, redirect, render_template

app = Flask(__name__)

modell = pickle.load(open('model.pkl','rb'))
target_names = ['setosa', 'versicolor', 'virginica']

@app.route("/")
def home():
    return render_template('index.html',prediction_txt="here")

@app.route("/predict",methods=["POST"])
def predict():
    features = [int(x) for x in request.form.values()]
    predict = modell.predict(np.array(features).reshape(1, -1))
    name=target_names[int(predict)]
    #name=predict
    return render_template('index.html',prediction_txt=str(name))

