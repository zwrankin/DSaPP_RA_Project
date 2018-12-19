import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load
from .utils import recode_tf_binary, map_categoricals
from .nlp import preprocess_phrase

DATA_DIR = '../data'
MODEL_DIR = '../models'
OUTCOME = 'fully_funded'


def load_data(data_dir=DATA_DIR, outcome=OUTCOME):
    df_projects = pd.read_csv(f'{data_dir}/projects.csv')
    df_projects = recode_tf_binary(df_projects)

    df_outcomes = pd.read_csv(f'{data_dir}/outcomes.csv')
    df_outcomes = recode_tf_binary(df_outcomes)

    df_essays = pd.read_csv(f'{data_dir}/essays.csv')
    df_essays.drop('teacher_acctid', axis=1, inplace=True)

    df = pd.merge(df_essays, df_projects, on='projectid')
    df = pd.merge(df, df_outcomes[['projectid', outcome]])

    df = preprocess_features(df)

    return df


def preprocess_features(df):
    """"Lazy feature engineering that avoids leakage, should be incorporated into pipeline"""
    # Price per student
    df['price_per_student'] = df.total_price_including_optional_support / df.students_reached
    # Make missing if infinity
    df.loc[df.students_reached == 0, 'price_per_student'] = np.NaN

    # Time
    df['date'] = pd.to_datetime(df.date_posted)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Map rare categoricals
    df['state_mapped'] = map_categoricals(df.school_state, noisy=False)
    df['city_mapped'] = map_categoricals(df.school_city, noisy=False)
    df['primary_focus_subject_mapped'] = map_categoricals(df.primary_focus_subject, noisy=False)

    return df


def process_nlp_data(words='title'):

    df = load_data()

    df_test = df.query('year == 2013')
    df_train = df.query('year < 2013')

    essays_train = preprocess_phrase(df_train[words])
    essays_test = preprocess_phrase(df_test[words])
    y_train = df_train[OUTCOME]
    y_test = df_test[OUTCOME]

    # Feature engineering
    cv = CountVectorizer(binary=True, stop_words='english')
    cv.fit(essays_train)
    dump(cv, f'{MODEL_DIR}/cv.joblib')

    X_train = cv.transform(essays_train)
    X_test = cv.transform(essays_test)

    # Save to processed data directory
    with open(f'{DATA_DIR}/processed/X_train', 'wb') as fp:
        pickle.dump(X_train, fp)
    with open(f'{DATA_DIR}/processed/X_test', 'wb') as fp:
        pickle.dump(X_test, fp)
    with open(f'{DATA_DIR}/processed/y_train', 'wb') as fp:
        pickle.dump(y_train, fp)
    with open(f'{DATA_DIR}/processed/y_test', 'wb') as fp:
        pickle.dump(y_test, fp)


def load_count_vectorizer(model_dir=MODEL_DIR):
    """Loads trained count vectorizer"""
    return load(f'{model_dir}/cv.joblib')


def load_nlp_data(data_dir=DATA_DIR):
    with open(f'{data_dir}/processed/X_train', 'rb') as fp:
        X_train = pickle.load(fp)
    with open(f'{data_dir}/processed/X_test', 'rb') as fp:
        X_test = pickle.load(fp)
    with open(f'{data_dir}/processed/y_train', 'rb') as fp:
        y_train = pickle.load(fp)
    with open(f'{data_dir}/processed/y_test', 'rb') as fp:
        y_test = pickle.load(fp)

    return X_train, X_test, y_train, y_test
