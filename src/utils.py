import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


def recode_tf_binary(df):
    """Recodes t/f to 0/1 dummies in anticipation of later ML models"""
    d = {'f': 0, 't': 1}
    for col in df.columns:
        if all(val in ['f', 't', np.nan] for val in df[col].unique().tolist()):
            df[col] = df[col].map(d)
    return df


def map_categoricals(feature:pd.Series, threshold=0.01, noisy=True):
    """Maps rare feature occurences to 'OTHER' if occurence is less than absolute or relative threshold """
    if threshold < 1:
        threshold *= len(feature)
    tabs = feature.value_counts()
    levels = tabs[tabs > threshold]
    if noisy:
        print(f'Reducing levels from {len(tabs)} to {len(levels)}')
    feature_encoded = feature.apply(lambda x: x if x in levels else 'OTHER')
    return feature_encoded


def pipeline_cv_score(model_pipeline, X, y, cv, n_jobs=-1):
    """Display cross-validation score of a pipeline given data X and y"""
    cv_score = cross_val_score(model_pipeline, X, y, cv=cv, n_jobs=n_jobs)
    print(f'Cross Validation Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')


def regularization_decay(frequency, k=20, f=10):
    "https://www.slideshare.net/0xdata/feature-engineering-83511751"
    return 1 / (1 + np.exp(-1*(frequency-k)/f))

class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    PRELIMINARY WAY OF DEALING WITH HIGH DIMENSIONALITY FEATURES
    """
    def __init__(self, weighted=True):
        self.weighted = weighted

    def fit(self, X, y, id_var):
        self.id_var = id_var
        df = pd.DataFrame({f'{id_var}':X[id_var], 'target':y})
        self.data = df.groupby(id_var).agg(['mean', 'count']).reset_index()
        self.data.columns = [id_var, f'{id_var}_mean', f'{id_var}_count']
        
        if self.weighted: 
            k = self.data[f'{id_var}_count'].median()
            self.weights = regularization_decay(self.data[f'{id_var}_count'])
        else: 
            self.weights = 1 
        
        self.global_mean = y.mean()
        self.data[f'{id_var}_mean'] = self.weights * self.data[f'{id_var}_mean'] + (1 - self.weights) * self.global_mean

    def transform(self, X):
        # TODO - how to account for ones without merge - fill missing with global mean? What about counts? 
        df = pd.merge(X, self.data, how='outer')
        fill_vals = {f'{self.id_var}_mean': self.global_mean, f'{self.id_var}_count': 0}
        df = df[[f'{self.id_var}_mean', f'{self.id_var}_count']].fillna(fill_vals)
        return df
        