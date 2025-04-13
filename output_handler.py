# output_handler.py - Updated with visualization support
"""
Enhanced output handler with visualization support.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class OutputHandler:
    """
    Enhanced output handler with visualization capabilities.
    """
    
    @staticmethod
    def write_dataframe(df: pd.DataFrame, filepath: str) -> None:
        """Write DataFrame to CSV with directory creation"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")
    
    @staticmethod
    def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: str):
        """Create model comparison visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # MAE comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='MAE', data=metrics_df)
        plt.title('Model Comparison by MAE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_mae_comparison.png'))
        plt.close()
        
        # R2 comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='R2', data=metrics_df)
        plt.title('Model Comparison by R²')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_r2_comparison.png'))
        plt.close()
    
    @staticmethod
    def plot_phone_durations(phone_stats: pd.DataFrame, output_dir: str):
        """Visualize phone duration statistics"""
        os.makedirs(output_dir, exist_ok=True)
        
        top_phones = phone_stats.sort_values('count', ascending=False).head(20)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='phone', y='avg_actual_duration', data=top_phones)
        plt.title('Average Duration by Phone (Top 20)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phone_duration_distribution.png'))
        plt.close()