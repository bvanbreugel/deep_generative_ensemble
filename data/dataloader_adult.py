from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def load_adult_census(as_frame=True, path='data/adult.csv'):
    df = pd.read_csv(path, encoding='latin-1')
    df[df==' ?'] = np.nan
    for col in ['workclass', 'occupation', 'native.country']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    # drop education because it is already encoded in education.num
    X, y = df.drop(['income', 'education'], axis=1), df['income']
    categorical = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    
    for feature in categorical:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
    
    if as_frame:
        X = pd.DataFrame(X)
        y = pd.Series(y)
    
    return X, y


if __name__ == '__main__':
    X, y = load_adult_census()
    print(X.head())
    print(y.head())
    print('Mean\n',X.mean(axis=0))
    print('Shape', X.shape)