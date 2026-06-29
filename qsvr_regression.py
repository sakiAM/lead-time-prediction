import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.algorithms.regressors import QSVR
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None
    from sklearn.metrics import mean_squared_error


DATASET_PATH = "DataCoSupplyChainDataset.csv"
TARGET_COLUMN = "Days for shipping (real)"
DATE_COLUMN = "shipping date (DateOrders)"
FEATURE_COLUMNS = [
    "Days for shipment (scheduled)",
    "Product Price",
    "shipping_day_of_week",
    "Order Item Quantity",
]
NUM_QUBITS = 4
TRAIN_SAMPLE_SIZE = 800
TEST_SAMPLE_SIZE = 200


def calculate_rmse(y_true, y_pred):
    if root_mean_squared_error is not None:
        return root_mean_squared_error(y_true, y_pred)
    return mean_squared_error(y_true, y_pred, squared=False)


def main():
    df = pd.read_csv(DATASET_PATH, encoding="latin1")
    algorithm_globals.random_seed = 42

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["shipping_day_of_week"] = df[DATE_COLUMN].dt.dayofweek

    model_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=model_columns)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train_sample = X_train.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    X_test_sample = X_test.sample(n=TEST_SAMPLE_SIZE, random_state=42)
    y_test_sample = y_test.loc[X_test_sample.index]

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = scaler.fit_transform(X_train_sample)
    X_test_scaled = scaler.transform(X_test_sample)

    feature_map = zz_feature_map(feature_dimension=NUM_QUBITS, reps=2)

    simulator = AerSimulator(method="statevector")
    sampler = SamplerV2(
        seed=42,
        options={
            "backend_options": simulator.options,
            "run_options": {"shots": None},
        },
    )
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
    )

    model = QSVR(quantum_kernel=quantum_kernel)
    model.fit(X_train_scaled, y_train_sample.to_numpy())

    y_pred = model.predict(X_test_scaled)
    rmse = calculate_rmse(y_test_sample, y_pred)
    mae = mean_absolute_error(y_test_sample, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
