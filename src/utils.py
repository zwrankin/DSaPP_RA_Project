import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

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
