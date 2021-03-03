from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello Flask on ML Model Deployment"

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    model = joblib.load('C:\\Users\\uxoriousghost\\Python Machine Learning\\ML Model Deploy on Flask\\model.pkl')
    to_predict_dict = request.form.to_dict()
    to_predict_list = list(to_predict_dict.values())
    to_predict_array = np.array(list(map(float, to_predict_list))).reshape(1, -1)
    prediction = model.predict(to_predict_array)
    return jsonify({'prediction': list(prediction)})

if __name__ == "__main__":
    app.run(debug = True)