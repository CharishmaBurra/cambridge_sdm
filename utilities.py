"""
utilities.py

Contains helper functions and classes for modeling, evaluation metrics,
and any general-purpose utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor


def compute_accuracy(actual_val: float, err_val: float) -> float:
    """
    Row-level accuracy = 1 - (err_val / actual_val), if actual_val != 0, else 0.
    """
    if actual_val and actual_val != 0:
        return 1.0 - (err_val / actual_val)
    return 0.0


class DurationModeler:
    """
    Class responsible for:
      - Creating and storing multiple regression models.
      - Training and evaluating them on native data.
      - Applying them to non-native data.
    """

    def __init__(self):
        """
        Initializes the list of scikit-learn regressors to train.
        """
        self.models = [
            ("LinearRegression",       LinearRegression()),
            ("DecisionTree",           DecisionTreeRegressor(random_state=42)),
            ("RandomForest",           RandomForestRegressor(n_estimators=100, random_state=42)),
            ("GradientBoosting",       GradientBoostingRegressor(random_state=42)),
            ("HistGradientBoosting",   HistGradientBoostingRegressor(random_state=42)),
            ("KNeighbors",             KNeighborsRegressor()),
        ]
        self.fitted_models = {}
        self.overall_metrics = []

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_predictions: pd.DataFrame
    ) -> None:
        """
        Train all models, evaluate them, and store row-level predictions + metrics.

        :param X_train: Training features (native phones).
        :param y_train: Training targets (durations).
        :param X_test: Test features (subset of native phones).
        :param y_test: Test targets (durations).
        :param test_predictions: DataFrame that will hold row-level predictions, errors, accuracy.
        """
        # For each model, fit, predict, compute metrics, and store results
        for model_name, model_obj in self.models:
            print(f"\n--- Training model: {model_name} ---")
            model_obj.fit(X_train, y_train)

            y_pred = model_obj.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)

            print(f"{model_name} -> MAE: {mae:.4f}, MSE: {mse:.4f}, R^2: {r2:.4f}")
            self.overall_metrics.append({
                "Model": model_name,
                "MAE": mae,
                "MSE": mse,
                "R2": r2
            })

            # Store the fitted model for later use
            self.fitted_models[model_name] = model_obj

            # Add columns for row-level predictions, error, accuracy
            pred_col = f"pred_{model_name}"
            err_col  = f"err_{model_name}"
            acc_col  = f"acc_{model_name}"

            test_predictions[pred_col] = y_pred
            test_predictions[err_col]  = abs(test_predictions['actual_duration'] - test_predictions[pred_col])
            test_predictions[acc_col]  = test_predictions.apply(
                lambda row: compute_accuracy(row['actual_duration'], row[err_col]),
                axis=1
            )

    def apply_to_non_native(
        self,
        non_native_df: pd.DataFrame,
        phone_col: str = "phone_encoded",
        actual_col: str = "actual_duration"
    ) -> pd.DataFrame:
        """
        Applies each fitted model to non-native data to produce predicted durations (and optionally errors/accuracy
        if actual durations are available).

        :param non_native_df: DataFrame with non-native data. Must have 'phone_encoded' column.
        :param phone_col: Column name for encoded phone IDs (default: phone_encoded).
        :param actual_col: Column name that holds actual duration for the non-native dataset (if present).
        :return: DataFrame with new columns for each model's predictions, errors, and accuracy.
        """
        if phone_col not in non_native_df.columns:
            print("\nNon-native data missing 'phone_encoded'. Cannot predict. Exiting.")
            return non_native_df

        X_non_native = non_native_df[[phone_col]].fillna(-1)

        # If we have a real 'duration' column, rename or confirm it
        if actual_col not in non_native_df.columns:
            print("\nNo ground-truth durations for non-native data. Errors/accuracy will be 0.")
            non_native_df[actual_col] = float('nan')

        for model_name, model_obj in self.fitted_models.items():
            pred_col = f"pred_{model_name}"
            err_col  = f"err_{model_name}"
            acc_col  = f"acc_{model_name}"

            y_pred = model_obj.predict(X_non_native)
            non_native_df[pred_col] = y_pred

            # If we do NOT actually have ground-truth durations, err & acc won't be meaningful
            if non_native_df[actual_col].notna().any():
                non_native_df[err_col] = abs(non_native_df[actual_col] - non_native_df[pred_col])
                non_native_df[acc_col] = non_native_df.apply(
                    lambda row: compute_accuracy(row[actual_col], row[err_col])
                                if pd.notna(row[actual_col])
                                else 0.0,
                    axis=1
                )
            else:
                non_native_df[err_col] = 0.0
                non_native_df[acc_col] = 0.0

        return non_native_df


def group_by_phone(
    df: pd.DataFrame,
    phone_col: str = "phone",
    actual_col: str = "actual_duration"
) -> pd.DataFrame:
    """
    Groups the test predictions DataFrame by phone, computing aggregated stats per phone.
    """
    model_pred_cols = [c for c in df.columns if c.startswith("pred_")]
    model_err_cols = [c for c in df.columns if c.startswith("err_")]
    model_acc_cols = [c for c in df.columns if c.startswith("acc_")]

    def aggregate_phone_metrics(grp: pd.DataFrame) -> pd.Series:
        result = {}
        result['count'] = len(grp)
        result['avg_actual_duration'] = grp[actual_col].mean()

        for pred_col in model_pred_cols:
            model_name = pred_col.replace("pred_", "")
            avg_pred = grp[pred_col].mean()
            result[f"avg_pred_{model_name}"] = avg_pred

        for err_col in model_err_cols:
            model_name = err_col.replace("err_", "")
            phone_mae = grp[err_col].mean()
            result[f"mae_{model_name}"] = phone_mae

        for acc_col in model_acc_cols:
            model_name = acc_col.replace("acc_", "")
            avg_acc = grp[acc_col].mean()
            result[f"avg_acc_{model_name}"] = avg_acc

        return pd.Series(result)

    return df.groupby(phone_col).apply(aggregate_phone_metrics).reset_index()
