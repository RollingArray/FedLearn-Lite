import os
import pandas as pd
import numpy as np
import tensorflow as tf
import flwr as fl
from model import build_model

CLIENT_ID = int(os.environ.get("CLIENT_ID", 1))
DATA_PATH = f"./data/client-data-{CLIENT_ID}.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    x = df[["x"]].values.astype(np.float32)
    y = df[["y"]].values.astype(np.float32)
    return x, y

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = build_model()
        self.x_train, self.y_train = load_data()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=5, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"mae": mae}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
