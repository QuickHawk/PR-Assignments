from flask import Flask, render_template, request
import pandas
import numpy
from io import BytesIO
from stats import (
    no_of_features,
    no_of_numerical_features,
    no_of_qualitative_features,
    find_stats_of_all_numerical_columns,
)

app = Flask(__name__)


def get_statistics(data: pandas.DataFrame):
    stats = {}

    stats["no_of_features"] = no_of_features(data)
    stats["no_of_numerical_features"] = no_of_numerical_features(data)
    stats["no_of_qualitative_features"] = no_of_qualitative_features(data)

    stats["skew"] = data.select_dtypes(include=[numpy.number]).skew().to_frame().to_html()
    stats["kurtosis"] = data.select_dtypes(include=[numpy.number]).kurtosis().to_frame().to_html()

    stats["other_stats"] = find_stats_of_all_numerical_columns(data).to_html()

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

        feature_data = data[data.columns.values[1:]]

        stats = get_statistics(feature_data)

        print(data.head().to_html())

        values = {"sample_data": data.head().to_html(), **stats}

        return render_template("result.html", **values)
    else:
        return render_template("error_page.html")


if __name__ == "__main__":
    app.run(debug=True)
