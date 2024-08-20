import numpy as np
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
    
    df=pd.read_csv(source_file).drop(labels=['customer_id'], axis=1)
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
    df=data.copy()

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

    print(f'auc_train: {round(auc_train, 4)}')
    print(f'auc_test: {round(auc_test, 4)}')
    
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



def compute_auc_per_category(model, X_train, y_train, X_test, y_test, cat_feature):
    """
    Считает ROC_AUC по указанной категории.

    Принимает:
        * model - модель
        * X_train - тренировочный сет предикторов
        * y_train - тренировочный сет целевой переменной
        * X_test - тестовый сет предикторов
        * y_test - тестовый сет целевой переменной
        * cat_feature - название категориального предиктора 

    Возвращает:
        * список, содержащий:
            - category - название категории
            - train_count - количество наблюдений в тренировочном сете
            - train_auc - ROC_AUC тренировочного сета
            - test_count - количество наблюдений в тестовом сете
            - test_auc - ROC_AUC тестового сета
    """
    unique_categories = X_train[cat_feature].unique()
    results = []

    for category in unique_categories:
        train_indices = X_train[cat_feature] == category
        test_indices = X_test[cat_feature] == category
        
        train_count = np.sum(train_indices)
        test_count = np.sum(test_indices)
        
        try:
            train_auc = metrics.roc_auc_score(y_train[train_indices], model.predict(X_train[train_indices]))
            test_auc = metrics.roc_auc_score(y_test[test_indices], model.predict(X_test[test_indices]))
        except ValueError:
            train_auc = np.nan
            test_auc = np.nan

        results.append([category, train_count, train_auc, test_count, test_auc])

    return results

def get_auc_per_category(model, X_train, y_train, X_test, y_test, cat_features):
    """
    Создаёт таблицу сравнений метрики ROC_AUC по категориям.

    Принимает:
        * model - модель
        * X_train - тренировочный сет предикторов
        * y_train - тренировочный сет целевой переменной
        * X_test - тестовый сет предикторов
        * y_test - тестовый сет целевой переменной
        * cat_features - список категориального предикторов
    
    Возвращает:
        * df - датафрейм
    """
    if type(cat_features)!=list:
        cat_features=[cat_features]
    
    df=pd.DataFrame()
    columns=['Category', 'Train Count', 'Train AUC', 'Test Count', 'Test AUC']
    
    for feature in cat_features:
        results = compute_auc_per_category(model, X_train, y_train, X_test, y_test, feature)
        df_results = (
            pd.DataFrame(results, columns=columns)
            .assign(feature=feature)
        )
        
        df_results['AUC Difference'] = df_results['Train AUC'] - df_results['Test AUC']
        df_results_sorted = (
            df_results
            .sort_values(by='AUC Difference', ascending=False)
            .drop(['AUC Difference'], axis=1)
        )

        df=pd.concat([df, df_results_sorted], ignore_index=True)
    return df



def compare_proportions(cat, train_set, test_set):
    """
    Считает пропорции категорий переданного предиктора в тренировочном и тестовом сете.

    Принимает:
        * cat - название категориального предиктора
        * train_set - тренировочный сет
        * test_set - тестовый сет
    
    Возвращает:
        * df - датафрейм
    """
    df=round(pd.concat([
        train_set[cat].value_counts(normalize=True),
        test_set[cat].value_counts(normalize=True)
    ], keys=['train', 'test'], axis=1), 2)
    return df



