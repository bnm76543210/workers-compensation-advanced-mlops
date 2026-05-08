"""
Загрузка и кэширование датасета Workers Compensation
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import streamlit as st
from src.logger import log

DATA_PATH = Path("data/workers_comp_raw.csv")

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        log("load_data: чтение локального CSV", path=str(DATA_PATH))
        df = pd.read_csv(DATA_PATH)
    else:
        log("load_data: запрос датасета OpenML id=42876")
        data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
        df = data.frame.copy()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        log("load_data: сырые данные получены и сохранены",
            rows=len(df), cols=df.shape[1], path=str(DATA_PATH))

    # Приводим числовые столбцы
    numeric_cols = ['Age', 'WeeklyPay', 'InitialCaseEstimate',
                    'UltimateIncurredClaimCost', 'HoursWorkedPerWeek',
                    'DaysWorkedPerWeek']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    nulls = df.isnull().sum().sum()
    log("load_data: готово", rows=len(df), cols=df.shape[1], nulls=nulls)
    return df

def get_feature_target(df):
    target = 'UltimateIncurredClaimCost'
    drop_cols = [target, 'ClaimDescription'] \
                if 'ClaimDescription' in df.columns else [target]
    X = df.drop(columns=drop_cols)
    y = df[target]
    return X, y
