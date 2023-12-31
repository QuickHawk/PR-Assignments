from flask import Flask, render_template, request, session, redirect
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
    confusion_matrix_plot_encoded,
    qual_features_probabilities,
    bayes_pred
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


@app.route("/select-columns", methods = ["GET", "POST"])
def select_columns():

    params = {}

    if "data" not in session or "data_orig" not in session:
        return render_template("error_page.html")
    
    data_orig = session["data_orig"]
    data = session["data"]

    available_columns = data_orig.columns.values[1:]
    params["available_columns"] = available_columns

    if "selected_columns" not in params:
        selected_columns = data.columns.values[1:]
        params["selected_columns"] = selected_columns

    if request.method == "POST":
        class_column = data_orig.columns.values[0]

        selected_columns = request.form.getlist("columns")
        params["selected_columns"] = selected_columns

        session["data"] = session["data_orig"][[class_column, *selected_columns]]
        session["selected_features"] = selected_columns

    return render_template("select_columns.html", **params)

@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        data_file = request.files["data"].read()
        data_bytes = BytesIO(data_file)
        data = pandas.read_csv(data_bytes)

        session["data"] = data
        session["data_orig"] = data

    return redirect("select-columns")

@app.route("/result", methods=["GET", "POST"])
def result_page():
   
    if "data" not in session:
        return render_template("error_page.html")
        
    data = session["data"]
    stats = get_statistics(data)
    values = {"sample_data": data.sample(5).to_html(), **stats}

    session["values"] = values

    return render_template("result.html", **values)

@app.route("/class-wise")
def class_wise_distribution():

    if "data" not in session:
        return render_template("error_page.html")

    data = session["data"]

    stats = get_statistics(data)
    session["values"] = stats

    return render_template("class_wise_distribution.html", **session["values"])

@app.route("/feature-wise")
def feature_wise_distribution():
    if "data" not in session:
        return render_template("error_page.html")

    data = session['data']
    feature_stats = feature_wise_stats(data)

    return render_template("feature_wise_distribution.html", feature_stats=feature_stats)

@app.route("/qualitative_features")
def qual_feat():
    if "data" not in session:
        return render_template("error_page.html")
    
    data = session["data"]
    qual_data_prob = qual_features_probabilities(data)

    return render_template("qual_features.html", qual_data = qual_data_prob)

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
    
    data = session["data"]
    print(data.columns.values)
    values["data"] = data

    if request.method == "POST":
        input_x_string = request.form["input"]

        input_x = pandas.read_csv(StringIO(input_x_string), header=None)
        input_x.columns = data.columns.values[1:]
        
        if not input_x.empty:
            
            qual_feat = qual_features_probabilities(data)
            results_num = predict_gaussian(data, input_x.select_dtypes(include=[numpy.number]))
            results_qual = bayes_pred(data, input_x.select_dtypes(exclude=[numpy.number]), qual_feat)

            print(results_num)
            print(results_qual)

            values["results_num"] = results_num
            values["results_qual"] = results_qual

        conf_matrix, list_of_classes = get_confusion_matrix(data, 0.3)
        img_data = confusion_matrix_plot_encoded(conf_matrix, list_of_classes)

        values["img_data"] = img_data

    # values["qualitative_prob"] = qual_features_probabilities(data)

    return render_template("predict.html", **values)

if __name__ == "__main__":
    app.run(debug=True)
