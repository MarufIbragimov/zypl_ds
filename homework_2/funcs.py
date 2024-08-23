import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

from sklearn.model_selection import train_test_split
from sklearn import metrics
from catboost import CatBoostClassifier, Pool

from sklearn.preprocessing import PolynomialFeatures
import featuretools as ft

import shap
from sklearn.feature_selection import RFECV

import category_encoders as ce
#from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier



seed=42

def get_bank_data():
    """
    Читает данные из файла 'Bank Customer Churn Prediction.csv'.
    Возвращает датафрейм.
    """
    source_file='data/Bank Customer Churn Prediction.csv'
    
    df=pd.read_csv(source_file).drop(labels=['customer_id'], axis=1)
    return df

#########################################################################################################################################################################################################

def split(data, target, test_size=.2, cat_encode=False, cat_cols=None,):
    """
    Делит данные на тренировочный и тестовый сеты.

    Принимает:
        * data - датасет
        * target - название целевой переменной
        * test_size - пропорция тестового сета
        * cat_encode - параметр указывающий следует ли кодировать категориальные столбцы
        * cat_cols - список категориальных столбцов
    Возвращает:
        * X_train - тренировочный сет предикторов
        * X_test - тестовый сет предикторов
        * y_train - тренировочный сет целевой переменной
        * y_test - тестовый сет целевой переменной
    """
    df=data.copy()
    
    X=df.drop(labels=[target], axis=1)
    y=df[target]

    if cat_encode:
        target_encoder = ce.TargetEncoder(cols=cat_cols)
        X = target_encoder.fit_transform(X, y)
     
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def make_classifier(depth=2, n_estimators=None):
    """
    Создаёт классификатор.

    Принимает:
        * depth - количество нод
        * n_estimator - количество деревьев
    """
    model=CatBoostClassifier(
        random_state=seed,
        depth=depth,
        n_estimators=n_estimators,
        eval_metric='AUC',
        verbose=0
    )

    return model


def train_catboost(train_pool, test_pool, type='classifier'):
    """
    Тренирует модель на базе алгоритмов CatBoost.

    Принимает:
        * train_pool - тренировочный сет в формате pool
        * test_pool - тестовый сет в формате pool
        * type - тип алгоритма (classifier или regressor)
    
    Возвращает:
        * model - тренированную модель
    """
    if type=='classifier':
        model=make_classifier()

    model.fit(train_pool, eval_set=test_pool, plot=True)
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    """
    Оценивает модель по метрике ROC_AUC.

    Принимает:
        * model - тренированную модель
        * train_pool - тренировочный сет предикторов в формате pool
        * test_pool - тестовый сет предикторов в формате pool
        * y_train - тренировочный сет целевой переменной
        * y_test - тестовый сет целевой переменной
    
    Возвращает:
        * auc_train - оценку ROC_AUC на тренировочном сете
        * auc_test - оценку ROC_AUC на тестовом сете
    """
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)

    auc_train=metrics.roc_auc_score(y_train, y_pred_train)
    auc_test=metrics.roc_auc_score(y_test, y_pred_test)

    return auc_train, auc_test


def build_model(data, cat_features, type='classifier'):
    """
    Создаёт модель предсказания оттока клиентов банка на основе алгоритма CatBoostClassifier.

    Принимает:
        * data - датафрейм содержащий данные банка
        * cat_features - список категориальных полей
        * type - тип модели

    Возвращает:
        * model - тренированная на данных модель
        * X_train - тренировочный сет предикторов, 
        * y_train - тренировочный сет целевой переменной, 
        * X_test - тестовый сет предикторов, 
        * y_test - тестовый сет целевой переменной, 
        * auc_train - ROC_AUC тренировочного сета, 
        * auc_test - ROC_AUC тестового сета
    """    

    X_train, X_test, y_train, y_test = split(data, 'churn')

    train_pool=Pool(data=X_train, label=y_train, cat_features=cat_features)
    test_pool=Pool(data=X_test, label=y_test, cat_features=cat_features)

    model=train_catboost(train_pool, test_pool, type)    

    auc_train, auc_test=evaluate(model, train_pool, test_pool, y_train, y_test)    

    print(f'auc_train: {round(auc_train, 4)}')
    print(f'auc_test: {round(auc_test, 4)}')
    
    return model, X_train, y_train, X_test, y_test, auc_train, auc_test

#########################################################################################################################################################################################################    

def get_feature_importance(model, feature_names):
    """
    Строит датафрейм важности признаков.

    Принимает:
        * model - тренированную модель
        * feature_names - названия предикторов
    
    Возвращает:
        * df - датафрейм с отсортированными по важности предикторами
    """
    df=pd.DataFrame({
        'feature':feature_names,
        'importance':model.get_feature_importance()
    }).sort_values(by=['importance'], ascending=False)

    return df


def plot_feature_importance(importance_df):
    """
    Создаёт диаграмму feature_importance.

    Принимает:
        * immportance_df - датафрейм с отранжированными по важности предикторами
    """
    sns.barplot(
        data=importance_df,
        y='feature',
        x='importance'
    );

#########################################################################################################################################################################################################

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

#########################################################################################################################################################################################################

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

#########################################################################################################################################################################################################

def print_in_sequence(data, column, subtitle):
    """
    Последовательно выводит на экран датафреймы разбитые по указанному полю.

    Принимает:
        * data - данные в виде датафрейма
        * column - поле, по которому нужно разбить датафрейм
        * subtitle - шаблон надписи, которую нужно отобразить над каждым сабсетом
    """
    for feature in data[column].unique():
        display(Markdown(subtitle.format(feature)))
        display(data.query(f"{column}==@feature"))
        print("\n" + "-" * 50 + "\n") 

#########################################################################################################################################################################################################

def add_polynomes(data, num_features):
    """
    Считает и добавляет полиномы в виде новых полей в переданный датасет.

    Принимает:
        * data - датафрейм с данными
        * num_features - список числовых столбцов

    Возвращает:
        * df - датафрейм с добавленными полиномиальными полями 
    """
    num_df = data[num_features].copy()

    poly = PolynomialFeatures(degree=4, include_bias=False, interaction_only=False)

    poly_features = poly.fit_transform(num_df)
    poly_feature_names = poly.get_feature_names_out(input_features=num_df.columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df = pd.concat([data.drop(num_features, axis=1), poly_df], axis=1)

    return df

#########################################################################################################################################################################################################

def add_aggregations(data, cat_features, cont_features):
    """
    Считает и добавляет аггрегированные значения по категориям в переданный датафрейм.

    Принимает:
        * data - датафрейм с данными
        * cat_features - список категориальных полей
        * cont_features - список полей с числовыми значениями

    Возвращает:
        * df - датафрейм с добавленными аггрегациями по категориям
    """
    df=data.copy()

    stats_dict={
        'mean':'mean',
        'std':'std',
        'median':lambda x: x.quantile(.5),
        'q1':lambda x: x.quantile(.25),
        'q3':lambda x: x.quantile(.25),
        'min':'min',
        'max':'max'
    }

    for cat in cat_features:
        for cont in cont_features:
            for k, v in stats_dict.items():
                df[f'{cont}_{k}']=df.groupby(cat, observed=True)[cont].transform(v)

    return df

#########################################################################################################################################################################################################

def add_some_math(data, cat_features):

    df=data.copy()

    es = ft.EntitySet(id = 'bank_churns')
    es = es.add_dataframe(
        dataframe_name='churns',
        dataframe = df.drop(labels=['churn'], axis=1),
        #variable_types = {x: ft.variable_types.Categorical for x in cat_features},
        #make_index = True,
        index = 'index'
    )

    trans_primitives = ['sine', 'cosine', 'natural_logarithm', 'square_root', 'add_numeric']

    features, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='churns',
        #logical_types={x:ft.variable_types.Categorical for x in cat_features},
        trans_primitives=trans_primitives
    )

    features=features.merge(
        right=df[['churn']],
        how='left',
        left_index=True,
        right_index=True
    )

    return features, feature_defs

#########################################################################################################################################################################################################

def plot_shap(model, model_name, X_train):
    """
    Визуализирует SHAP (SHapley Additive exPlanations) summary.

    Принимает:
        * model - тренированная модель
        * model_name - название модели
        * X_train - тренировочный сет
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(10, 30))
    plt.title(f'SHAP plot for {model_name}', loc='center', fontdict={'fontsize':20}, pad=20)
    shap.summary_plot(shap_values, X_train)

#########################################################################################################################################################################################################

def select_features_recursively(X, y, cv=3, min_features_to_select=1):
    rfecv = RFECV(
        estimator=make_classifier(n_estimators=100),
        step=1,
        cv=cv,
        scoring="roc_auc",
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    
    rfecv.fit(X, y)
    
    selected_features=rfecv.get_feature_names_out().tolist()

    return selected_features

#########################################################################################################################################################################################################

def build_stacked_model(data, cat_cols, estimators):
    """
    Создаёт модель предсказания оттока клиентов банка с применением стеккинга.

    Принимает:
        * data - датафрейм содержащий данные банка
        * cat_cols - список категориальных полей
        * estimators - словарь содержащий базовые модели

    Возвращает:
        * model - тренированная на данных модель
        * X_train - тренировочный сет предикторов, 
        * y_train - тренировочный сет целевой переменной, 
        * X_test - тестовый сет предикторов, 
        * y_test - тестовый сет целевой переменной, 
        * roc_auc_train - ROC_AUC тренировочного сета, 
        * roc_auc_test - ROC_AUC тестового сета
    """
    X_train, X_test, y_train, y_test = split(data, target='churn', cat_cols=cat_cols, cat_encode=True)
    
    for model in estimators.values():
        model.fit(X_train, y_train)
        auc_train, auc_test = evaluate(model, X_train, X_test, y_train, y_test)
        print(f"{model.__class__.__name__} AUC train: {auc_train:.4f}")
        print(f"{model.__class__.__name__} AUC test: {auc_test:.4f}")

    catboost_model = make_classifier(depth=3, n_estimators=100)

    stacking_model = StackingClassifier(
        estimators=[(k, v) for k,v in estimators.items()],
        final_estimator=catboost_model
    )

    stacking_model.fit(X_train, y_train)

    roc_auc_train, roc_auc_test = evaluate(stacking_model, X_train, X_test, y_train, y_test)

    print(f'\nroc_auc_train: {round(auc_train, 4)}')
    print(f'roc_auc_test: {round(auc_test, 4)}')

    return stacking_model, X_train, y_train, X_test, y_test, roc_auc_train, roc_auc_test