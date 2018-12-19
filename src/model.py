from sklearn.linear_model import Ridge, LogisticRegression
from joblib import dump, load

from .data import MODEL_DIR
from .data import load_nlp_data

def fit_nlp_model():
    X_train, X_test, y_train, y_test = load_nlp_data()

    # Fit
    # model = Ridge(alpha=0.5, random_state=0)
    model = LogisticRegression(solver='lbfgs', random_state=0)
    model.fit(X_train, y_train)

    # Save model
    dump(model, f'{MODEL_DIR}/nlp_model.joblib')

    return model


def load_nlp_model():
    return load(f'{MODEL_DIR}/nlp_model.joblib')
