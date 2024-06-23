import joblib
from sklearn.ensemble import RandomForestClassifier


def train(df):
    forest_df = df.drop(
        columns=["Filepath", "Timestamp", "Text", "Time", "Who answered"]).astype(float)
    params = {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 4,
              'min_samples_split': 10, 'n_estimators': 300}
    forest = RandomForestClassifier(**params)
    forest.fit(forest_df.drop(columns='Successful'), df['Successful'])
    joblib.dump(forest, 'random_forest_model.pkl')


def predict(df):
    forest = joblib.load('random_forest_model.pkl')
    forest_df = df.drop(
        columns=["Filepath", "Timestamp", "Text", "Time"]).astype(float)
    return forest.predict(forest_df)
