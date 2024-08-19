import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from catboost import CatBoostClassifier, Pool


seed=42

def get_bank_data():
    """
    Читает данные из файла 'Bank Customer Churn Prediction.csv'.
    Возвращает датафрейм.
    """
    source_file='data/Bank Customer Churn Prediction.csv'
    
    df=pd.read_csv(source_file)
    return df

def build_churn_model(data, cat_features):
    """
    Создаёт модель предсказания оттока клиентов банка на основе алгоритма CatBoostClassifier.

    Принимает:
        * data - датафрейм содержащий данные банка
        * cat_features - список категориальных полей

    Возвращает:
        * model - тренированная на данных модель
        * X_train - тренировочный сет предикторов, 
        * y_train - тренировочный сет целевой переменной, 
        * X_test - тестовый сет предикторов, 
        * y_test - тестовый сет целевой переменной, 
        * auc_train - ROC_AUC тренировочного сета, 
        * auc_test - ROC_AUC тестового сета
    """    
    target='churn'
    df=data.copy().drop(labels=['customer_id'], axis=1)

    X=df.drop(labels=[target], axis=1)
    y=df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)

    train_pool=Pool(data=X_train, label=y_train, cat_features=cat_features)
    test_pool=Pool(data=X_test, label=y_test, cat_features=cat_features)

    model=CatBoostClassifier(
        random_state=seed,
        depth=2,
        eval_metric='AUC',
        verbose=0
    )

    model.fit(train_pool, eval_set=test_pool, plot=True)

    y_pred_train=model.predict(train_pool)
    y_pred_test=model.predict(test_pool)

    auc_train=metrics.roc_auc_score(y_train, y_pred_train)
    auc_test=metrics.roc_auc_score(y_test, y_pred_test)

    return model, X_train, y_train, X_test, y_test, auc_train, auc_test

def plot_feature_importance(model, feature_names):
    """
    Создаёт диаграмму feature_importance.

    Принимает:
        * model - модель
        * feature_names - список предикторов
    """
    feature_importance = model.get_feature_importance()
    sorted_idx = feature_importance.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[sorted_idx], feature_importance[sorted_idx], align='center')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Plot')

    plt.show()