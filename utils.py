# utils.py
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    x = df[['x']].values
    y = df[['y']].values
    return x, y
