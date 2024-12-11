import numpy as np
from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

with open("DT_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("data_pipeline.pkl", 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)

cols = ['TIME', 'LATITUDE', 'LONGITUDE', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH', 'SPEEDING',
        'AG_DRIV', 'DAY_OF_WEEK', 'ROAD_CLASS', 'DISTRICT', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE']


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/result", methods=["POST"])
def result():
    time = np.array([request.form['time'].replace(':', '')])
    longitude = np.array([request.form['longi']])
    latitude = np.array([request.form['lati']])
    pedestrian = np.array([request.form['pedestrian']])
    cyclist = np.array([request.form['cyclist']])
    automobile = np.array([request.form['automobile']])
    truck = np.array([request.form['truck']])
    transit = np.array([request.form['transit']])
    emergency = np.array([request.form['emergency']])
    speeding = np.array([request.form['speeding']])
    aggresive = np.array([request.form['aggresive']])
    week = np.array([request.form['week']])
    roadClass = np.array([request.form['roadClass']])
    district = np.array([request.form['district']])
    traffic = np.array([request.form['traffic']])
    visibility = np.array([request.form['visibility']])
    light = np.array([request.form['light']])
    roadCondition = np.array([request.form['roadCondition']])
    impactType = np.array([request.form['impactType']])
    involvementType = np.array([request.form['involvementType']])

    final = np.concatenate([time, longitude, latitude, pedestrian, cyclist, automobile, truck, transit, emergency, speeding,
                           aggresive, week, roadClass, district,  traffic, visibility, light, roadCondition, impactType, involvementType])
    final = np.array(final)
    data = pd.DataFrame([final], columns=cols)
    data_transformed = pipeline.transform(data)
    prediction = model.predict(data_transformed)
    return render_template("result.html", prediction='Non-Fatal' if prediction == 0 else 'Fatal')


if __name__ == "__main__":
    app.run(debug=True, port=5000)
