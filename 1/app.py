from flask import Flask, render_template, request, session
import pandas
from flask_session import Session
import numpy
from io import BytesIO, StringIO
from stats import (
    no_of_features,
    no_of_numerical_features,
    no_of_qualitative_features,
    find_stats_of_all_numerical_columns,
    predict_gaussian
)

app = Flask(__name__)
app.secret_key = "Testing 123"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def get_statistics(data: pandas.DataFrame):
    stats = {}

    stats["no_of_features"] = no_of_features(data)
    stats["no_of_numerical_features"] = no_of_numerical_features(data)
    stats["no_of_qualitative_features"] = no_of_qualitative_features(data)

    stats["stats"] = find_stats_of_all_numerical_columns(data)

    stats["numerical_column_names"] = data.iloc[:,1:].select_dtypes(include=[numpy.number]).columns.values.tolist(),


    # print(stats)

    return stats


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result_page():
    if request.method == "POST":
        data_file = request.files["data"].read()
        data_bytes = BytesIO(data_file)
        data = pandas.read_csv(data_bytes)

        if "data" in session:
            session.pop("data")
        session["data"] = data

    elif "data" in session:
        data = session["data"]
    else:
        return render_template("error_page.html")
        
    stats = get_statistics(data)
    values = {"sample_data": data.head().to_html(), **stats}

    if "values" in session:
        session.pop("values")

    session["values"] = values

    return render_template("result.html", **values)

@app.route("/class-wise")
def class_wise_distribution():
    if "values" in session:
        return render_template("class_wise_distribution.html", **session["values"])
    else:
        return render_template("error_page.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    values = {**session["values"]}

    # print("values", values)

    if "data" not in session:
        return render_template("error_page.html")

    if request.method == "POST":
        data = session["data"]
        input_x_string = request.form["input"]

        input_x = pandas.read_csv(StringIO(input_x_string), header=None)

        if not input_x.empty:
            results = predict_gaussian(data, input_x)
            values["results"] = results

    return render_template("predict.html", **values)

if __name__ == "__main__":
    app.run(debug=True)
