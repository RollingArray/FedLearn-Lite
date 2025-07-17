# FedLearn-Lite: Federated Learning for Linear Regression

A lightweight yet powerful federated learning simulation using Flower and TensorFlow — where multiple clients collaboratively train a shared global model without ever sharing their raw data.

> 📌 Perfect for demonstrating how decentralized learning can effectively learn a simple linear relationship: `y = 2x + 1`

---

## 🚀 Overview

`FedLearn-Lite` simulates a **realistic federated learning setup** across multiple clients. Each client owns a subset of data and trains a local model. A central server performs **Federated Averaging (FedAvg)** to aggregate the weights into a global model.

The project demonstrates:

- How **federated learning** works in a practical client-server architecture.
- How to split and distribute data across clients to preserve privacy.
- How a global model can learn a perfect linear mapping `y = 2x + 1` without any data centralization.
- Evaluation of global model after multiple rounds of learning.

---

## 🧠 Technologies Used

- [Flower](https://flower.dev) — Federated Learning Framework
- [TensorFlow](https://tensorflow.org) — Deep Learning Engine
- Python 3.x

---

## 🗂️ Project Structure

```text
FedLearn-Lite/
│
├── data/                     # Client-specific datasets
│   ├── client-data-1.csv
│   ├── client-data-2.csv
│   └── client-data-3.csv
│
├── weights/                  # Saved client model weights
│
├── model/                    # Saved global models after each round
│
├── plots/                    # Output visualizations (Actual vs Predicted)
│
├── model.py                  # Model architecture (Keras)
├── generate_data.py          # Script to generate and split data for clients
├── client.py                 # Federated client logic
├── server.py                 # Flower server with FedAvg strategy
├── test_model.py             # Evaluate and visualize final model
└── README.md
