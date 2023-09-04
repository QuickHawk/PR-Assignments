import pandas
import os
import numpy

def no_of_features(data: pandas.DataFrame) -> int:
    return len(data.columns.values)

def no_of_numerical_features(data: pandas.DataFrame) -> int:
    t = data.select_dtypes(include=[numpy.number])
    print(t.columns.values)
    return len(t.columns.values)

def no_of_qualitative_features(data: pandas.DataFrame) -> int:
    t = data.select_dtypes(exclude=[numpy.number])
    print(t.columns.values)
    return len(t.columns.values)

def find_stats_of_all_numerical_columns(data: pandas.DataFrame) -> pandas.DataFrame:
    return data.describe()

if __name__ == "__main__":
    print(os.path.abspath("."))
    data = pandas.read_csv("./csv/titanic_data.csv")
    print(data.columns)
    print(no_of_features(data))
    print(no_of_numerical_features(data))
    print(no_of_qualitative_features(data))
    print(data.describe())