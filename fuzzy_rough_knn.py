import joblib
from sklearn.metrics import accuracy_score

from frlearn.base import probabilities_from_scores, select_class
from frlearn.classifiers import FRNN
from frlearn.feature_preprocessors import RangeNormaliser

model_path = 'fuzzy_rough_knn_model.pkl'

def train(df):
    X = df.drop(columns=["Filepath", "Time", "Successful", "Who answered"]).astype(float).values
    y = df["Successful"].astype(float).values
    clf = FRNN(preprocessors=(RangeNormaliser(), ), upper_weights=None, upper_k=14, lower_k=14)
    model = clf(X, y)
    joblib.dump(model, model_path)


def predict(df):
    model = joblib.load(model_path)
    X = df.drop(columns=["Filepath", "Time", "Successful", "Who answered"]).astype(float).values
    scores = model(X)
    return select_class(scores)
