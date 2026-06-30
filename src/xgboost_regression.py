import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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
    "Order Item Quantity",
    "Product Price",
    "shipping_day_of_week",
    "is_weekend",
    "Shipping Mode",
]


def calculate_rmse(y_true, y_pred):
    if root_mean_squared_error is not None:
        return root_mean_squared_error(y_true, y_pred)
    return mean_squared_error(y_true, y_pred, squared=False)


def main():
    df = pd.read_csv(DATASET_PATH, encoding="latin1")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["shipping_day_of_week"] = df[DATE_COLUMN].dt.dayofweek
    df["is_weekend"] = df["shipping_day_of_week"].isin([5, 6]).astype(int)

    model_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=model_columns)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X = pd.get_dummies(X, columns=["Shipping Mode"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = calculate_rmse(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print("\nFeature Importances:")

    feature_importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    for feature_name, importance in feature_importances.items():
        print(f"{feature_name}: {importance:.6f}")


if __name__ == "__main__":
    main()
