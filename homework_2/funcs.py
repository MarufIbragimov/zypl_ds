import pandas as pd

def get_bank_data():
    """
    Читает данные из файла 'Bank Customer Churn Prediction.csv'.
    Возвращает датафрейм.
    """
    source_file='data/Bank Customer Churn Prediction.csv'
    
    df=pd.read_csv(source_file)
    return df