from catboost import CatBoostClassifier


def train(df):
    params = {'depth': 4, 'iterations': 200, 'l2_leaf_reg': 5,
              'learning_rate': 0.05}
    model = CatBoostClassifier(**params)
    model.fit(df.drop(columns='Successful'), df['Successful'])
    model.save_model('cat_boost_model.cbm')


def predict(df):
    model = CatBoostClassifier()
    model.load_model('cat_boost_model.cbm')
    return model.predict(df.drop(columns='Filepath'))
