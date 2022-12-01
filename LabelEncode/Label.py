import sklearn
from sklearn import preprocessing

def Label_conv(data):
    le = preprocessing.LabelEncoder()
    # Assigning numerical values and storing in another column
    data['Label'] = le.fit_transform(data)
    df=data['Label'].copy()
    
    return df