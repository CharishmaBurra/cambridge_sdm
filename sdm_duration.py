import os
import glob
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def extract_phoneme_data(json_file):
    """
    Parses a JSON alignment file to extract a list of records.
    Each record is a dictionary with keys 'phone' and 'duration'.
    Returns an empty list if the file cannot be processed.
    """
    if not json_file.lower().endswith('.json'):
        print(f"SKIP (not .json): {json_file}")
        return []

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"WARNING: Could not parse {json_file}. Skipping. Error: {e}")
        return []

    # Check for expected structure
    if not isinstance(data, dict) or 'result' not in data or not isinstance(data['result'], dict):
        print(f"WARNING: {json_file} does not have the expected structure. Skipping.")
        return []

    records = []
    segments = data['result'].get('segments', [])
    for segment in segments:
        words = segment.get('words', [])
        for word in words:
            phones = word.get('phones', [])
            for phone in phones:
                phone_label = phone.get('phone')
                duration = phone.get('duration')
                if phone_label is not None and duration is not None:
                    records.append({'phone': phone_label, 'duration': duration})
    print(f"DEBUG: Extracted {len(records)} records from {json_file}")
    return records

def load_all_data(directory):
    """
    Recursively reads all JSON files in the given directory and returns a list
    of phoneme records (each is a dict with keys 'phone' and 'duration').
    """
    pattern = os.path.join(directory, '**', '*.json')
    json_files = glob.glob(pattern, recursive=True)
    all_records = []
    for jf in json_files:
        records = extract_phoneme_data(jf)
        all_records.extend(records)
    return all_records

def compute_baseline(training_records, test_records):
    """
    Computes baseline predictions based on the per-phone mean duration from training_records.
    For each test record, if its phone is found in training_records, the mean duration is used;
    otherwise, the overall mean duration is used.
    
    :param training_records: list of dicts with keys 'phone' and 'duration' (training data)
    :param test_records: list of dicts with key 'phone' (test data)
    :return: tuple (predictions, phone_mean, overall_mean)
             - predictions: list of predicted durations for each test record.
             - phone_mean: dictionary mapping phone to mean duration.
             - overall_mean: overall mean duration across all training records.
    """
    # Compute mean duration for each phone in training data
    phone_durations = {}
    for record in training_records:
        phone = record['phone']
        duration = record['duration']
        if phone not in phone_durations:
            phone_durations[phone] = []
        phone_durations[phone].append(duration)
    
    phone_mean = {}
    for phone, durations in phone_durations.items():
        phone_mean[phone] = sum(durations) / len(durations)
    
    # Compute overall mean duration
    all_durations = [rec['duration'] for rec in training_records]
    overall_mean = sum(all_durations) / len(all_durations) if all_durations else 0.0

    # Predict duration for each test record
    predictions = []
    for record in test_records:
        phone = record.get('phone')
        prediction = phone_mean.get(phone, overall_mean)
        predictions.append(prediction)
    
    return predictions, phone_mean, overall_mean

def evaluate_baseline(test_records, predictions):
    """
    Evaluates baseline predictions using MAE, MSE, and R^2.
    
    :param test_records: list of dicts with key 'duration' as the ground truth.
    :param predictions: list of predicted durations corresponding to test_records.
    :return: (mae, mse, r2)
    """
    actual = [record['duration'] for record in test_records]
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    return mae, mse, r2

def visualize_predictions(test_records, predictions):
    """
    Creates a scatter plot of actual vs. predicted durations.
    Also plots an ideal (y=x) line for reference.
    
    :param test_records: list of dicts with key 'duration' (actual durations).
    :param predictions: list of predicted durations.
    """
    actual = [record['duration'] for record in test_records]
    
    plt.figure()
    plt.scatter(actual, predictions, alpha=0.5)
    plt.xlabel("Actual Duration (seconds)")
    plt.ylabel("Predicted Duration (seconds)")
    plt.title("Baseline: Actual vs Predicted Durations")
    
    # Plot ideal line
    min_val = min(actual)
    max_val = max(actual)
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.tight_layout()
    plt.show()

def main():
    # Set paths for training and test data directories.

    training_dir = "/Volumes/New Folder/Python Practice/cambridge/data/american_english"
    test_dir = "/Volumes/New Folder/Python Practice/cambridge/data/other_english"
    
    # Load data from directories (handles subdirectories)
    print("Loading training data...")
    training_records = load_all_data(training_dir)
    print("Loading test data...")
    test_records = load_all_data(test_dir)
    
    print(f"Training records loaded: {len(training_records)}")
    print(f"Test records loaded: {len(test_records)}")
    
    # Compute baseline predictions
    predictions, phone_mean, overall_mean = compute_baseline(training_records, test_records)
    
    print("\n--- Baseline Predictions ---")
    print("Per-phone mean durations (from training):")
    for phone, mean_val in phone_mean.items():
        print(f"{phone}: {mean_val:.4f}")
    print(f"\nOverall Mean Duration: {overall_mean:.4f}")
    
    print("\nSample Predictions for test records:")
    for idx, pred in enumerate(predictions[:10]):  # Show first 10 predictions
        print(f"Record {idx + 1}: Predicted Duration = {pred:.4f}")
    
    # Evaluate predictions if ground truth is available
    mae, mse, r2 = evaluate_baseline(test_records, predictions)
    print("\n--- Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    # Visualize the results
    visualize_predictions(test_records, predictions)

if __name__ == '__main__':
    main()
