import pandas as pd
import matplotlib.pyplot as plt

def visualize_overall_metrics(metrics_csv_path: str):
    """
    Reads the overall_metrics.csv file, then plots
    MAE, MSE, and R^2 for each model as separate bar charts.
    """
    # 1) Load
    df = pd.read_csv(metrics_csv_path)

    # 2) Bar chart for MAE
    plt.figure()
    plt.bar(df['Model'], df['MAE'])
    plt.title("Mean Absolute Error by Model")
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()  
    plt.show()

    # 3) Bar chart for MSE
    plt.figure()
    plt.bar(df['Model'], df['MSE'])
    plt.title("Mean Squared Error by Model")
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 4) Bar chart for R^2
    plt.figure()
    plt.bar(df['Model'], df['R2'])
    plt.title("R^2 Score by Model")
    plt.xlabel("Model")
    plt.ylabel("R^2")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def visualize_phone_level_stats(phone_stats_csv_path: str):
    """
    Reads phone_level_stats.csv and plots:
    1) A scatter comparing avg_actual_duration vs. avg_pred_{Model}
    2) A bar chart of MAE for a selected model

    Adjust model_name to whichever model you want to examine.
    """
    # 1) Load
    df = pd.read_csv(phone_stats_csv_path)

    # Choose one model to highlight. Example: "RandomForest"
    model_name = "RandomForest"
    avg_pred_col = f"avg_pred_{model_name}"
    mae_col = f"mae_{model_name}"

    # 2) Scatter Plot: actual vs. predicted (for the chosen model)
    plt.figure()
    plt.scatter(df['avg_actual_duration'], df[avg_pred_col])
    plt.title(f"Phone-Level: Actual vs. Predicted Duration ({model_name})")
    plt.xlabel("Average Actual Duration")
    plt.ylabel(f"Average Predicted Duration ({model_name})")
    plt.tight_layout()
    plt.show()

    # 3) Bar Chart: top 20 phones by count, and their phone-level MAE
    df_top = df.sort_values('count', ascending=False).head(20)

    plt.figure()
    plt.bar(df_top['phone'], df_top[mae_col])
    plt.title(f"MAE by Phone (Top 20) - {model_name}")
    plt.xlabel("Phone")
    plt.ylabel("MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    """
    Example driver function to visualize both overall_metrics.csv
    and phone_level_stats.csv. Adapt paths to match your real output files.
    """
    overall_metrics_path = "/overall_metrics.csv"
    phone_stats_path = "/phone_level_stats.csv"

    visualize_overall_metrics(overall_metrics_path)
    visualize_phone_level_stats(phone_stats_path)

if __name__ == "__main__":
    main()
