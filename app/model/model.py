import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import FeatureUnion
import joblib


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


class OHEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.columns = []

    def fit(self, X, y=None):
        self.columns = [col for col in pd.get_dummies(X, prefix=self.key).columns]
        return self

    def transform(self, X):
        X = pd.get_dummies(X, prefix=self.key)
        test_columns = [col for col in X.columns]
        for col_ in self.columns:
            if col_ not in test_columns:
                X[col_] = 0
        return X[self.columns]

class NumericPower(BaseEstimator, TransformerMixin):
    def __init__(self, key, p = 2):
        self.key = key
        self.columns = []
        self.p = p+1

    def fit(self, X, y=None):
        B = [self.key + str(i) for i in range(1, self.p)]
        self.columns = B + ['log']
        return self

    def transform(self, X):
        Xp = X.values.reshape(-1,1)
        for i in range(2, self.p):
            Xp = np.hstack([Xp,(X.values.reshape(-1,1) ** i).astype(float)])

        Xp = np.hstack([Xp, np.log(X.values.reshape(-1,1) + 1).astype(float)])
        B = pd.DataFrame(data = Xp, index = X.index,columns =[self.columns] )
        return B[self.columns]


def get_pipeline():
    num_cols = ['route_minutes', 'total_area', 'rooms']
    cat_cols = ['metro', 'okrug']

    final_transformers = list()

    for cat_col in cat_cols:
        cat_transformer = Pipeline([
                    ('selector', FeatureSelector(column=cat_col)),
                    ('ohe', OHEEncoder(key=cat_col))
                ])
        final_transformers.append((cat_col, cat_transformer))

    for num_col in num_cols:
        cont_transformer = Pipeline([
                    ('selector', NumberSelector(key=num_col)),
                    ('power', NumericPower(key=num_col, p=3)),
                    ('scale', StandardScaler())
                ])
        final_transformers.append((num_col, cont_transformer))

    feats = FeatureUnion(final_transformers)
    pipeline = Pipeline([
        ('features', feats),
        ('regressor', CatBoostRegressor(iterations=1000, max_depth=10, random_state=42, silent=True)),
    ])

    return pipeline


def fit_pipeline(X_train, y_train, pipeline, save_model=False):
    pipeline.fit(X_train, y_train)
    if save_model:
        joblib.dump(pipeline, 'model.pkl')

    return pipeline


if __name__ == '__main__':
    df = pd.read_csv("../data/moscow_estate.csv", names=['okrug', 'metro', 'distance_from_center', 'route_minutes', 'total_area', 'rooms', 'price'])
    pipe = get_pipeline()
    pipe = fit_pipeline(df.drop(['price'], axis=1), df['price'], pipe, save_model=True)
    preds = pipe.predict(pd.DataFrame({'metro': 'Народное ополчение',
                           'okrug': 'СЗАО',
                           'distance_from_center': 9.8,
                           'route_minutes': 10,
                           'rooms': 2,
                           'total_area': 45.0}, index=[0]))
    true = 12000000
    print(preds[0])
    print(abs(np.ceil(true-preds[0])))
