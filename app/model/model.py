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
                    ('scale', StandardScaler())
                ])
        final_transformers.append((num_col, cont_transformer))

    feats = FeatureUnion(final_transformers)
    pipeline = Pipeline([
        ('features', feats),
        ('classifier', CatBoostRegressor(iterations=2000, max_depth=10, learning_rate=1, silent=True, random_state=42)),
    ])

    return pipeline


def fit_pipeline(X_train, y_train, pipeline, save_model=False):
    pipeline.fit(X_train, y_train)
    if save_model:
        joblib.dump(pipeline, 'model.pkl')

    return pipeline


if __name__ == '__main__':
    df = pd.read_csv("../data/moscow_estate.csv", names=['okrug', 'metro', 'route_minutes', 'total_area', 'rooms', 'price'])
    pipe = get_pipeline()
    pipe = fit_pipeline(df.drop(['price'], axis=1), df['price'], pipe, save_model=True)
    preds = pipe.predict(pd.DataFrame({'metro': 'Октябрьское Поле',
                           'okrug': 'СЗАО',
                           'route_minutes': 8,
                           'rooms': 2,
                           'total_area': 51.0}, index=[0]))
    print(preds[0])
