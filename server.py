import flwr as fl
from model import build_model
import numpy as np
import tensorflow as tf

# Optional: Evaluation function to run after each round
def get_eval_fn():
    model = build_model()
    
    def evaluate(server_round, parameters, config):
        model.set_weights(parameters)

        # Save the global model
        model_path = f"./model/global_model_round_{server_round}.h5"
        model.save(model_path)
        print(f"💾 [Round {server_round}] Global model saved to {model_path}")

        # Evaluate on test point
        x = np.array([[4.0]], dtype=np.float32)
        y_true = np.array([[9.0]], dtype=np.float32)  # Expected y = 2x + 1 = 9

        loss, mae = model.evaluate(x, y_true, verbose=0)
        y_pred = model.predict(x, verbose=0)
        print(f"📈 [Round {server_round}] Prediction for x=4: y={y_pred[0][0]:.4f}")

        print(f"📊 [Round {server_round}] Loss: {loss:.4f}, MAE: {mae:.4f}")
        return loss, {"mae": mae}
    
    return evaluate


# Define strategy with enhanced logging
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    evaluate_fn=get_eval_fn(),
    on_fit_config_fn=lambda rnd: {"round": rnd}
)

if __name__ == "__main__":
    print("🚀 Starting Federated Learning Server...")
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    print("✅ Server has shut down.")
