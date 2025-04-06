"""
output_handler.py

Responsible for writing final results (predictions, metrics, etc.) to disk.
"""

import pandas as pd
import os


class OutputHandler:
    """
    A class to centralize all CSV or file output logic.
    """

    @staticmethod
    def write_dataframe(df: pd.DataFrame, filepath: str) -> None:
        """
        Writes a DataFrame to the specified CSV file, creating directories if needed.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Wrote DataFrame to {filepath}")
