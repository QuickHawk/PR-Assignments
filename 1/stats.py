import pandas
import numpy
from scipy.stats import multivariate_normal 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot
import matplotlib
from io import BytesIO
import seaborn
import base64
import math

matplotlib.use('agg')

def no_of_features(data: pandas.DataFrame) -> int:
    return data.columns[1:].values

def no_of_numerical_features(data: pandas.DataFrame) -> int:
    t = data.iloc[:, 1:].select_dtypes(include=[numpy.number])
    return t.columns.values

def no_of_qualitative_features(data: pandas.DataFrame) -> int:
    t = data.iloc[:, 1:].select_dtypes(exclude=[numpy.number])
    return t.columns.values

def find_stats_of_all_numerical_columns(df: pandas.DataFrame):
    stats = {}
    
    _class_column_name = df.columns.values[0]
    _class_list = df.iloc[:, 0].unique()

    for _class in _class_list:
        temp_data = df[df[_class_column_name] == _class].iloc[:, 1:]

        skew_data = temp_data.select_dtypes(include=[numpy.number]).skew().to_frame().T
        skew_data.index = ["Skew"]

        kurtosis_data = temp_data.select_dtypes(include=[numpy.number]).kurtosis().to_frame().T
        kurtosis_data.index = ["Kurtosis"]
        
        desc_data = temp_data.describe().copy()
        
        desc_data = pandas.concat([desc_data, skew_data, kurtosis_data])
        

        print(desc_data)

        stats[_class] = {
            "other": desc_data.to_html(),
            "covariance": temp_data.select_dtypes(include=[numpy.number]).cov(ddof=0).to_html(),
            "correlation": temp_data.select_dtypes(include=[numpy.number]).corr().to_html(),
        }

    return stats

def pdf(data: pandas.DataFrame, x):

    _class_list = data.iloc[:, 0].unique()
    results = {}

    for _class in _class_list:
        class_data = data[data.iloc[:, 0] == _class]
        class_numeric_data = class_data.iloc[:, 1:].select_dtypes(include=[numpy.number])

        class_cov = class_numeric_data.cov(ddof=0)
        class_mean = class_numeric_data.mean()

        y = multivariate_normal.pdf(x, class_mean, class_cov, allow_singular=True)

        results[_class] = y
    
    return results

def feature_wise_stats(data: pandas.DataFrame):
    num_data = data.iloc[:,1:].select_dtypes(include=[numpy.number])
    
    label = data.columns.values[0]
    features = num_data.columns.values

    feature_stats = {}

    for feat in features:
        feature_stats[feat] = data[[label, feat]].groupby(label).describe()

    return feature_stats

def box_plot_data(data: pandas.DataFrame):
    
    fig, ax = pyplot.subplots()

    data.boxplot(ax=ax, by=data.columns.values[0])

    temp_file = BytesIO()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    fig.savefig(temp_file, format='png')
    temp_file.seek(0)

    image_encoded = base64.b64encode(temp_file.getvalue()).decode()

    return image_encoded

def confusion_matrix_plot_data(conf_matrix, list_of_classes):
    
    fig, ax = pyplot.subplots()

    seaborn.heatmap(conf_matrix, xticklabels=list_of_classes, yticklabels=list_of_classes, annot=True, ax=ax)
    # t = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list_of_classes)
    # t.plot(ax=ax)

    temp_file = BytesIO()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    fig.savefig(temp_file, format='png')
    temp_file.seek(0)

    image_encoded = base64.b64encode(temp_file.getvalue()).decode()

    return image_encoded

def image_data(data: pandas.DataFrame):
    
    labels = data.columns.values[0]
    numerical_features = data.iloc[:,1:].select_dtypes(include=[numpy.number]).columns.values

    encodings = []

    for feature in numerical_features:
        encodings.append(box_plot_data(data[[labels, feature]]))

    return encodings
    
def confusion_matrix_data(data: pandas.DataFrame, split_ratio = 0.2):

    train_test_n = math.floor(data.shape[0] * split_ratio)

    shuffled_data = data.sample(frac=1)

    train_data = shuffled_data.iloc[:train_test_n]
    test_data = shuffled_data.iloc[train_test_n:]

    list_of_classes = data.iloc[:, 0].unique().tolist()

    test_x = test_data.iloc[:, 1:].select_dtypes(include=[numpy.number])
    actual_y = test_data.iloc[:, 0]

    predicted_y = numpy.zeros((test_x.shape[0], 3))

    label_column_name = data.columns.values[0]

    for idx in range(len(list_of_classes)):
            
        _class = list_of_classes[idx]

        train_x = train_data[train_data[label_column_name] == _class].iloc[:, 1:].select_dtypes(include=[numpy.number])
        train_y = train_data.iloc[:, 0]

        train_mean = train_x.mean()
        train_cov = train_x.cov(ddof=0)

        predicted_y[:, idx] = multivariate_normal.pdf(test_x, train_mean, train_cov, allow_singular=True)

    actual_y = [list_of_classes.index(row) for row in actual_y]
    predicted_y = numpy.argmax(predicted_y, axis=1).tolist()

    conf_matrix = confusion_matrix(actual_y, predicted_y)

    return conf_matrix, list_of_classes

def qual_features_probabilities(data: pandas.DataFrame):
    
    label_column_name = data.columns.values[0]

    qual_feat_list = data.iloc[:,1:].select_dtypes(exclude=[numpy.number]).columns.values
    _class_list = data.iloc[:,0].unique()

    prob_data = {}

    for feat in qual_feat_list:
        list_of_values = data[feat].unique()
        t = pandas.DataFrame(index=_class_list, columns=list_of_values)
        for _class in _class_list:
            _class_data = data[data[label_column_name] == _class]
            for val in list_of_values:
                _temp_data = _class_data[_class_data[feat] == val]
                prob = len(_temp_data) / len(_class_data)
                
                t.loc[_class, val] = prob

        prob_data[feat] = t

    return prob_data

def bayes_pred(data: pandas.DataFrame, input_x: pandas.DataFrame, qual_features_probability: dict[str, pandas.DataFrame]):
    
    denom = 0
    result = {}

    prob_class = {}

    list_of_classes = data.iloc[:, 0].unique()
    label_column_name = data.columns.values[0]

    for _class in list_of_classes:
        p = data[data[label_column_name] == _class].shape[0] / data.shape[0]
        prob_class[_class] = p
    
    for _class in list_of_classes:
        prod = 1
        for _feat in input_x.columns.values:
            val = input_x[_feat]
            p = qual_features_probability[_feat].loc[_class, val].iloc[0]
            prod *= p

        prod *= prob_class[_class]
        denom += prod

    for _class in list_of_classes:
        num = 1
        for _feat in input_x.columns.values:
            val = input_x[_feat]
            p = qual_features_probability[_feat].loc[_class, val].iloc[0]
            num *= p

        num *= prob_class[_class]
        result[_class] = num/denom

    return result

    