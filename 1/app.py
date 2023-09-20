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
    predict_gaussian,
    feature_wise_stats,
    image_encodings,
    get_confusion_matrix,
    confusion_matrix_plot_encoded
)

app = Flask(__name__)
app.secret_key = "Testing 123"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def get_statistics(data: pandas.DataFrame):
    stats = {}

    nFeatures = no_of_features(data)
    nNumericalFeatures = no_of_numerical_features(data)
    nQualitativeFeatures = no_of_qualitative_features(data)

    stats["list_of_features"] = nFeatures
    stats["no_of_features"] = len(nFeatures)
    stats["list_of_numerical_features"] = nNumericalFeatures
    stats["no_of_numerical_features"] = len(nNumericalFeatures)
    stats["list_of_qualitative_features"] = nQualitativeFeatures
    stats["no_of_qualitative_features"] = len(nQualitativeFeatures)

    stats["labels"] = data.iloc[:,0].unique().astype(str).tolist()
    stats["n_labels"] = len(stats["labels"])

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
    values = {"sample_data": data.sample(5).to_html(), **stats}

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

@app.route("/feature-wise")
def feature_wise_distribution():
    if "data" not in session:
        return render_template("error_page.html")

    data = session['data']
    feature_stats = feature_wise_stats(data)

    return render_template("feature_wise_distribution.html", feature_stats=feature_stats)

@app.route("/plots")
def boxplots():
    if "data" not in session:
        return render_template("error_page.html")

    data = session['data']
    encodings = image_encodings(data)

    return render_template('boxplots.html', encodings = encodings)

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    values = {**session["values"]}

    if "data" not in session:
        return render_template("error_page.html")

    if request.method == "POST":
        data = session["data"]
        input_x_string = request.form["input"]

        input_x = pandas.read_csv(StringIO(input_x_string), header=None)

        if not input_x.empty:
            results = predict_gaussian(data, input_x)
            values["results"] = results

        conf_matrix, list_of_classes = get_confusion_matrix(data, 0.3)
        img_data = confusion_matrix_plot_encoded(conf_matrix, list_of_classes)

        values["img_data"] = img_data

    return render_template("predict.html", **values)

if __name__ == "__main__":
    app.run(debug=True)
