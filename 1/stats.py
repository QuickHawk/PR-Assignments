import pandas
import os
import numpy

def no_of_features(data: pandas.DataFrame) -> int:
    return len(data.columns[1:].values)

def no_of_numerical_features(data: pandas.DataFrame) -> int:
    t = data.iloc[:, 1:].select_dtypes(include=[numpy.number])
    print(t.columns.values)
    return len(t.columns.values)

def no_of_qualitative_features(data: pandas.DataFrame) -> int:
    t = data.iloc[:, 1:].select_dtypes(exclude=[numpy.number])
    print(t.columns.values)
    return len(t.columns.values)

def find_stats_of_all_numerical_columns(data: pandas.DataFrame):
    stats = {}
    
    label_column_name = data.columns.values[0]
    list_of_labels = data.iloc[:, 0].unique()
    
    for label in list_of_labels:
        temp_data = data[data[label_column_name] == label]
        stats[label] = {
            "other": temp_data.describe().to_html(),
            "skew": temp_data.select_dtypes(include=[numpy.number]).skew().to_frame().to_html(),
            "kurtosis": temp_data.select_dtypes(include=[numpy.number]).kurtosis().to_frame().to_html()
        }

    return stats

if __name__ == "__main__":
    print(os.path.abspath("."))
    data = pandas.read_csv("./csv/titanic_data.csv")
    print(data.columns)
    print(no_of_features(data))
    print(no_of_numerical_features(data))
    print(no_of_qualitative_features(data))
    print(data.describe())