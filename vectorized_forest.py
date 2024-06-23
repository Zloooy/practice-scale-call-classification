import joblib
from sklearn.ensemble import RandomForestClassifier



forest_columns = [*map(str, range(256)), "Successful"]
model_file = 'vectorized_random_forest_model.pkl'
def train(df):
    forest_df = df[forest_columns].astype(float)
    params = {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 4,
              'min_samples_split': 10, 'n_estimators': 300}
    forest = RandomForestClassifier(**params)
    forest.fit(forest_df.drop(columns='Successful'), df['Successful'])
    joblib.dump(forest, model_file)


def predict(df):
    forest = joblib.load(model_file)
    forest_df = df[forest_columns].astype(float)
    return forest.predict(forest_df)
