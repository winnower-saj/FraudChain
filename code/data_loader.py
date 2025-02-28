import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../elliptic_bitcoin_dataset/")

def load_data():
    df_features = pd.read_csv(os.path.join(DATASET_PATH, "elliptic_txs_features.csv"), header=None)
    df_classes = pd.read_csv(os.path.join(DATASET_PATH, "elliptic_txs_classes.csv"))

    df_features["tx_id"] = df_classes["txId"]
    df_features["class"] = df_classes["class"]

    df_features["class"] = df_features["class"].replace({"unknown": -1, "1": 1, "2": 0}) # 1-> Illicit; 2-> Fraud; 3->Unknown
    df_features = df_features[df_features["class"] != -1]

    scaler = StandardScaler()
    feature_columns = df_features.columns[:-2]
    df_features[feature_columns] = scaler.fit_transform(df_features[feature_columns])

    return df_features
