import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None
    from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


DATASET_PATH = "DataCoSupplyChainDataset.csv"
FEATURE_COLUMNS = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Product Price",
]
TARGET_COLUMN = "Days for shipping (real)"


def main():
    df = pd.read_csv(DATASET_PATH, encoding="latin1")
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

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_test, y_pred)
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
