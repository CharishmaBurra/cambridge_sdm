Here’s a README file template for your project based on the provided code:

---

# Subword Unit Duration Modeling

This project focuses on creating machine learning models to predict the duration of subword units (phonemes) for both native and non-native English speakers. The project involves multiple steps, including loading datasets, training multiple regression models, evaluating them, and saving the results in CSV files for further analysis.

## Project Structure

The project is organized into the following modules:

1. **`config.py`** - Contains configuration constants and parameters, such as directory paths and output file locations.
2. **`input_handler.py`** - Handles the loading and parsing of JSON alignment files into Pandas DataFrames.
3. **`main.py`** - The main entry point that ties everything together. This script loads the datasets, trains the models, evaluates the performance, and saves the results.
4. **`output_handler.py`** - Responsible for writing the output DataFrames (predictions, metrics, etc.) to CSV files.
5. **`utilities.py`** - Contains helper functions for modeling, metrics evaluation, and general-purpose utilities like data splitting, regression models, etc.

### Data Flow

1. **Data Loading**: The `InputHandler` class in `input_handler.py` loads phoneme-level data from JSON alignment files located in the directories specified in `config.py`.
2. **Model Training**: The `DurationModeler` class in `utilities.py` trains multiple regression models (e.g., Linear Regression, Random Forest) on the native dataset.
3. **Evaluation**: The models are evaluated on both the training and test sets, and evaluation metrics like MAE (Mean Absolute Error), MSE (Mean Squared Error), and R² (R-squared) are computed.
4. **Predictions**: Once trained, the models are used to make predictions on a non-native dataset.
5. **Output**: The results (predictions and metrics) are saved to CSV files using `OutputHandler` in `output_handler.py`.

## Requirements

Before running the project, make sure the following libraries are installed:

- `pandas >= 1.0.0`
- `scikit-learn >= 0.22`
- `matplotlib >= 3.0.0`

You can install them using `pip` by running:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/subword-unit-duration-modeling.git
```

### 2. Install dependencies:
```bash
cd subword-unit-duration-modeling
pip install -r requirements.txt
```

### 3. Update the paths in `config.py`:
Ensure that the directory paths in `config.py` point to the correct locations of your input data (native and non-native datasets) and output directories.

```python
LOCAL_NATIVE_DIR = "/path/to/your/native_data"
LOCAL_NON_NATIVE_DIR = "/path/to/your/non_native_data"
```

### 4. Run the main script:
Once your directories are set, you can run the main script to start the model training, evaluation, and output generation.

```bash
python main.py
```

## Output

The following output CSV files will be generated:

- **`test_level_predictions.csv`**: Contains the test-level predictions for all models.
- **`phone_level_stats.csv`**: Contains aggregated statistics at the phone level (e.g., average prediction, MAE, accuracy).
- **`overall_metrics.csv`**: Contains the evaluation metrics for each model (MAE, MSE, R²).
- **`non_native_predictions_all_models2.csv`**: Contains the predictions for non-native data.

## Visualizations

You can visualize the performance of the models using the provided functions in `utilities.py`. For example:

- **Overall Model Metrics**: Plots bar charts for MAE, MSE, and R² for each model.
- **Phone-Level Stats**: Plots a scatter plot comparing the actual vs. predicted duration and bar charts for MAE by phone for a selected model.

To generate visualizations:

1. Make sure the `overall_metrics.csv` and `phone_level_stats.csv` files are available.
2. Run the following in `utilities.py`:

```bash
python utilities.py
```

## Explanation of Key Components

### DurationModeler Class
- Trains multiple regression models (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, etc.).
- Evaluates the models on the test set and stores metrics like MAE, MSE, and R².
- Uses these models to make predictions on non-native datasets.

### Grouping and Aggregating by Phone
The `group_by_phone` function in `utilities.py` groups the test predictions by phone and calculates aggregated statistics, such as average actual duration, average predicted duration, and MAE.

## Troubleshooting

- If the native dataset is empty or doesn't contain the required columns (`phone` and `duration`), the script will exit early with an error message.
- Ensure that the `phone_encoded` column is present in both native and non-native datasets to make predictions.

