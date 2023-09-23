import pandas
import numpy

from matplotlib import pyplot

data = pandas.read_csv("csv/titanic_data.csv")

numerical_feature_data = data.select_dtypes(include=[numpy.number])
print(data.head())
print(data.shape)
label = data.columns.values[0]
features = numerical_feature_data.columns.values[1:]
for feat in features:
    print()
    print(data[[label, feat]].groupby(label).describe())

# data[[label, 'Age']].boxplot(by=label)
# pyplot.show()