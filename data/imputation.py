# -*- coding: utf-8 -*-
import numpy as np
from sklearn.impute import SimpleImputer,KNNImputer

def Simimputer(data, method):
    method2option = {'Mean': 'mean', 'Median': 'median', 'Most Frequent': 'most_frequent'}[method]
    imp = SimpleImputer(missing_values=np.nan, strategy=method2option)
    print(f"Successfully fill the missing values with the {method2option} value "
          f"of each feature column respectively.")
    return imp.fit_transform(data)

def knnimputer(data):
    imp = KNNImputer(missing_values=np.nan, add_indicator=True)
    print(f"Successfully fill the missing values with the knn value "
          f"of each feature column respectively.")
    return imp.fit_transform(data)

def String_ch(data):
    df2= data.replace('<0.03', 0)
    data = df2.replace(np.nan, 0.0)
    data=data.replace({-0.01: 0.0})
    return data