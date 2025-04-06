"""
main.py

Entry point for the Subword Unit Duration Modeling. Ties together all modules:
 - configuration
 - input handling
 - utilities for modeling
 - output handling
and produces the same results as the original script.
"""

import pandas as pd

from config import (
    LOCAL_NATIVE_DIR,
    LOCAL_NON_NATIVE_DIR,
    TEST_LEVEL_OUTPUT_CSV,
    PHONE_LEVEL_OUTPUT_CSV,
    OVERALL_METRICS_CSV,
    NON_NATIVE_OUTPUT_CSV
)
from input_handler import InputHandler
from utilities import DurationModeler, group_by_phone
from output_handler import OutputHandler


def main():
    """
    Main entry point of the program.
    """
    # 1) Load data
    native_df = InputHandler.load_dataset(LOCAL_NATIVE_DIR)
    non_native_df = InputHandler.load_dataset(LOCAL_NON_NATIVE_DIR)

    print(f"\nNative dataset total records: {native_df.shape[0]}")
    print(f"Non-native dataset total records: {non_native_df.shape[0]}")

    if native_df.empty:
        print("ERROR: Native dataset is empty. Cannot train a model.")
        return

    required_cols = {'phone', 'duration'}
    if not required_cols.issubset(native_df.columns):
        missing = required_cols - set(native_df.columns)
        print(f"ERROR: Native dataset missing columns: {missing}")
        return

    # 2) Phone -> index map
    all_phones = pd.concat([native_df['phone'], non_native_df['phone']], ignore_index=True)
    unique_phones = all_phones.dropna().unique()
    phone_to_idx = {phone: idx for idx, phone in enumerate(unique_phones)}

    # 3) Encode phones
    native_df['phone_encoded'] = native_df['phone'].map(phone_to_idx)
    non_native_df['phone_encoded'] = non_native_df['phone'].map(phone_to_idx)

    # 4) Prepare training data
    X_native = native_df[['phone_encoded']]
    y_native = native_df['duration']

    # 5) Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_native, y_native, test_size=0.2, random_state=42
    )

    # Build a DataFrame to hold row-level test predictions
    test_predictions = X_test.copy()
    test_predictions['actual_duration'] = y_test.values
    inv_map = {v: k for k, v in phone_to_idx.items()}
    test_predictions['phone'] = test_predictions['phone_encoded'].map(inv_map)

    # 6) Create the DurationModeler and train multiple models
    modeler = DurationModeler()
    modeler.train_and_evaluate(X_train, y_train, X_test, y_test, test_predictions)

    # 7) Save the test-level predictions
    OutputHandler.write_dataframe(test_predictions, TEST_LEVEL_OUTPUT_CSV)

    # 8) Group by phone and save phone-level stats
    phone_grouped = group_by_phone(test_predictions)
    OutputHandler.write_dataframe(phone_grouped, PHONE_LEVEL_OUTPUT_CSV)

    # 9) Save overall metrics
    overall_df = pd.DataFrame(modeler.overall_metrics)
    OutputHandler.write_dataframe(overall_df, OVERALL_METRICS_CSV)

    # 10) Non-native predictions for all models
    if non_native_df.empty:
        print("\nNon-native dataset is empty. Skipping multi-model predictions.")
        return

    # If 'duration' is available, rename it to actual_duration or keep as is
    if 'duration' in non_native_df.columns:
        non_native_df.rename(columns={'duration': 'actual_duration'}, inplace=True)
    else:
        non_native_df['actual_duration'] = float('nan')

    if 'phone_encoded' not in non_native_df.columns:
        print("\nNon-native data missing 'phone_encoded'. Cannot predict. Exiting.")
        return

    non_native_preds = modeler.apply_to_non_native(non_native_df)
    OutputHandler.write_dataframe(non_native_preds, NON_NATIVE_OUTPUT_CSV)


if __name__ == "__main__":
    main()
