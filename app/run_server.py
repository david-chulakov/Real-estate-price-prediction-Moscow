import joblib
from flask import Flask, render_template, request
import pandas as pd
import pickle
from model import model
from model.model import FeatureSelector, NumberSelector, NumericPower, OHEEncoder
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def start():
    if request.method == 'POST':
        metro = request.form['metro']
        way = request.form['way']
        minutes = int(request.form['minutes'])
        living_area = int(request.form['living_area'])
        kitchen_area = int(request.form['kitchen_area'])
        total_area = int(request.form['total_area'])

        df = pd.DataFrame({'metro': metro,
                           'way': way,
                           'minutes': minutes,
                           'living_area': living_area,
                           'kitchen_area': kitchen_area,
                           'total_area': total_area}, index=[0])

        model = joblib.load('model/model.pkl')
        prediction = model.predict(df)

        return render_template("prediction.html", prediction=prediction[0])
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)