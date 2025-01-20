# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def handle_missing_values(train):
    train['education'] = train['education'].fillna(train['education'].mode()[0])
    train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].mean())
    return train

def casefolding(train):
    for col in train.select_dtypes(include='object').columns:
        train[col] = train[col].str.lower()
    return train

def encode_categorical_data(train):
    categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel']
    label_encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        train[column] = encoder.fit_transform(train[column])
        label_encoders[column] = encoder
    return train, label_encoders

def normalize_data(train):
    numerical_cols = ['age', 'length_of_service', 'avg_training_score', 'previous_year_rating']
    scaler = StandardScaler()
    train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
    return train, scaler

def feature_engineering(train, test):
    train['sum_metric'] = train['awards_won?'] + train['previous_year_rating']
    test['sum_metric'] = test['awards_won?'] + test['previous_year_rating']
    train['total_score'] = train['avg_training_score'] * train['no_of_trainings']
    test['total_score'] = test['avg_training_score'] * test['no_of_trainings']
    return train, test

def drop_unnecessary_columns(train, test):
    train = train.drop(['region', 'employee_id'], axis=1)
    test = test.drop(['region', 'employee_id'], axis=1)
    return train, test

def balance_data(train):
    y = train['is_promoted']
    x = train.drop(['is_promoted'], axis=1)
    x_resample, y_resample = RandomOverSampler(sampling_strategy=1).fit_resample(x, y.values.ravel())
    return x_resample, y_resample

def preprocess_data(train_path, test_path):
    train, test = load_data(train_path, test_path)
    train = handle_missing_values(train)
    train = casefolding(train)
    train, label_encoders = encode_categorical_data(train)
    train, scaler = normalize_data(train)
    train, test = feature_engineering(train, test)
    train, test = drop_unnecessary_columns(train, test)
    x_resample, y_resample = balance_data(train)
    return x_resample, y_resample, test, label_encoders, scaler