import pandas
import numpy
from scipy.stats import multivariate_normal 

def no_of_features(data: pandas.DataFrame) -> int:
    return len(data.columns[1:].values)

def no_of_numerical_features(data: pandas.DataFrame) -> int:
    t = data.iloc[:, 1:].select_dtypes(include=[numpy.number])
    # print(t.columns.values)
    return len(t.columns.values)

def no_of_qualitative_features(data: pandas.DataFrame) -> int:
    t = data.iloc[:, 1:].select_dtypes(exclude=[numpy.number])
    # print(t.columns.values)
    return len(t.columns.values)

def find_stats_of_all_numerical_columns(data: pandas.DataFrame):
    stats = {}
    
    label_column_name = data.columns.values[0]
    list_of_labels = data.iloc[:, 0].unique()

    for label in list_of_labels:
        temp_data = data[data[label_column_name] == label].iloc[:, 1:]
        stats[label] = {
            "other": temp_data.describe().to_html(),
            "skew": temp_data.select_dtypes(include=[numpy.number]).skew().to_frame().to_html(),
            "kurtosis": temp_data.select_dtypes(include=[numpy.number]).kurtosis().to_frame().to_html(),
            "covariance": temp_data.select_dtypes(include=[numpy.number]).cov(ddof=0).to_html(),
            "correlation": temp_data.select_dtypes(include=[numpy.number]).corr().to_html(),
        }

    return stats

def predict_gaussian(data: pandas.DataFrame, x):

    list_of_classes = data.iloc[:, 0].unique()
    results = {}

    for _class in list_of_classes:
        class_data = data[data.iloc[:, 0] == _class]
        class_numeric_data = class_data.select_dtypes(include=[numpy.number])

        class_cov = class_numeric_data.cov(ddof=0)
        class_mean = class_numeric_data.mean()

        print(x.to_numpy())

        y = multivariate_normal.pdf(x, class_mean, class_cov)

        results[_class] = y
    
    return results
