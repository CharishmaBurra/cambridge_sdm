"""
Enhanced Duration Modeling Pipeline with Improved Neural Network and Additional Metrics
"""

import os
import time
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, 
                           mean_squared_error, 
                           r2_score,
                           explained_variance_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from input_handler import EnhancedInputHandler
from output_handler import OutputHandler
from tensorflow.keras.layers import LayerNormalization, Add

from config import (
    LOCAL_NATIVE_DIR,
    LOCAL_NON_NATIVE_DIR,
    TEST_LEVEL_OUTPUT_CSV,
    PHONE_LEVEL_OUTPUT_CSV,
    OVERALL_METRICS_CSV,
    NON_NATIVE_OUTPUT_CSV,
    ERROR_ANALYSIS_DIR
)

def prepare_features(df, is_training=True):
    """Prepares feature matrix with proper encoding and validation"""
    if df.empty:
        return pd.DataFrame(), None, {}

    # Feature encoding
    unique_phones = df['phone'].unique()
    phone_to_idx = {phone: idx for idx, phone in enumerate(unique_phones)}
    df['phone_encoded'] = df['phone'].map(phone_to_idx)
    
    if 'phone_bigram' in df.columns:
        unique_bigrams = df['phone_bigram'].unique()
        bigram_to_idx = {bigram: idx for idx, bigram in enumerate(unique_bigrams)}
        df['phone_bigram_encoded'] = df['phone_bigram'].map(bigram_to_idx)
    
    # Feature selection with validation
    feature_sets = {
        'numerical': [
            'position_in_word', 'word_length', 'duration_ratio',
            'position_ratio', 'position_in_utterance', 'utterance_length',
            'position_ratio_utterance'
        ],
        'categorical': [
            'is_vowel', 'is_silence', 'is_consonant', 
            'is_voiced', 'is_plosive', 'is_fricative',
            'is_nasal', 'has_stress', 'is_word_initial',
            'is_word_final', 'prev_is_vowel', 'next_is_vowel'
        ],
        'encoded': [
            'phone_encoded',
            'phone_bigram_encoded' if 'phone_bigram_encoded' in df.columns else None
        ]
    }
    
    available_features = [
        f for features in feature_sets.values() 
        for f in features 
        if f in df.columns and f is not None
    ]
    
    print(f"Using {len(available_features)} features")
    
    # Prepare final data with type validation
    X = df[available_features].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(-1)
    
    y = df['duration'].copy() if is_training and 'duration' in df.columns else None
    
    return X, y, phone_to_idx

def train_optimized_models(X_train, y_train, X_test, y_test, test_predictions):
    """Enhanced training implementation with improved neural network and input scaling"""
    print("\nTraining optimized models...")
    start_time = time.time()
    models = {}
    feature_importances = {}

    # 1. Random Forest with feature importance
    print("\n[1/3] Training Random Forest...")
    rf_model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42
        )
    )
    rf_model.fit(X_train, y_train)
    test_predictions['RandomForest_pred'] = rf_model.predict(X_test)
    models['RandomForest'] = rf_model

    # Store feature importances
    if hasattr(rf_model.steps[-1][1], 'feature_importances_'):
        feature_importances['RandomForest'] = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': rf_model.steps[-1][1].feature_importances_
        }).sort_values('Importance', ascending=False)

    print(f"Completed in {time.time()-start_time:.1f}s")

    # 2. Gradient Boosting with early stopping
    print("\n[2/3] Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=5
    )
    gb_model.fit(X_train, y_train)
    test_predictions['GradientBoosting_pred'] = gb_model.predict(X_test)
    models['GradientBoosting'] = gb_model

    if hasattr(gb_model, 'feature_importances_'):
        feature_importances['GradientBoosting'] = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': gb_model.feature_importances_
        }).sort_values('Importance', ascending=False)

    print(f"Completed in {time.time()-start_time:.1f}s")

    # 3. Enhanced Neural Network with scaling
    print("\n[3/3] Training Neural Network...")

    # Scale inputs
    nn_scaler_X = StandardScaler()
    X_train_nn = nn_scaler_X.fit_transform(X_train)
    X_test_nn = nn_scaler_X.transform(X_test)

    # Scale target
    nn_scaler_y = StandardScaler()
    y_train_nn = nn_scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # Train model
    nn_model = make_enhanced_nn(X_train_nn.shape[1])
    history = nn_model.fit(
        X_train_nn, y_train_nn,
        epochs=100,
        batch_size=256,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ],
        verbose=1
    )

    # Predict and inverse scale
    nn_preds_scaled = nn_model.predict(X_test_nn).flatten()
    nn_preds = nn_scaler_y.inverse_transform(nn_preds_scaled.reshape(-1, 1)).flatten()

    test_predictions['NeuralNetwork_pred'] = nn_preds
    models['NeuralNetwork'] = nn_model

    # Optionally save scalers if you want to use the model later
    import joblib
    joblib.dump(nn_scaler_X, os.path.join(ERROR_ANALYSIS_DIR, 'nn_input_scaler.pkl'))
    joblib.dump(nn_scaler_y, os.path.join(ERROR_ANALYSIS_DIR, 'nn_target_scaler.pkl'))

    print(f"Completed in {time.time()-start_time:.1f}s")

    return models, feature_importances


from tensorflow.keras.layers import Add

def make_enhanced_nn(input_dim):
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=(input_dim,))

    # Block 1
    x = Dense(256, activation='swish', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Block 2 - Match output dim with residual
    x_skip = x  # shape (None, 256)
    x = Dense(256, activation='swish')(x)  # now also (None, 256)
    x = Add()([x, x_skip])  # ✅ now valid
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Block 3
    x = Dense(64, activation='swish')(x)
    x = BatchNormalization()(x)

    output = Dense(1)(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss='huber')
    return model



def analyze_results(test_predictions, models, feature_importances):
    """Enhanced results analysis with feature importance"""
    results = {}
    
    # Calculate metrics for each model
    metrics = []
    for name in models:
        pred_col = f'{name}_pred'
        if pred_col not in test_predictions.columns:
            print(f"Warning: Missing predictions for {name}")
            continue
            
        y_pred = test_predictions[pred_col]
        y_true = test_predictions['actual_duration']
        
        metrics.append({
            'Model': name,
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'Explained Variance': explained_variance_score(y_true, y_pred)
        })
        
        # Store predictions
        results[name] = {
            'predictions': y_pred,
            'actual': y_true
        }
    
    # Feature importance analysis
    for model_name, importance_df in feature_importances.items():
        print(f"\nTop 10 Features for {model_name}:")
        print(importance_df.head(10).to_string(index=False))
        
        # Save feature importance
        OutputHandler.write_dataframe(
            importance_df, 
            os.path.join(ERROR_ANALYSIS_DIR, f'{model_name.lower()}_feature_importance.csv')
        )
    
    return pd.DataFrame(metrics), results

def main():
    """Main execution pipeline with enhanced features"""
    try:
        gc.collect()  # Clean memory before starting
        print("===== Enhanced Duration Modeling Pipeline =====")
        
        # 1. Data Loading
        print("\n[1/4] Loading and processing data...")
        handler = EnhancedInputHandler()
        native_df = handler.load_dataset(LOCAL_NATIVE_DIR)
        
        if native_df.empty:
            print("Error: No valid data loaded")
            return
        
        print(f"Loaded {len(native_df)} records")

        # 2. Feature Preparation
        print("\n[2/4] Preparing features...")
        X, y, _ = prepare_features(native_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3. Model Training
        print("\n[3/4] Training models...")
        test_predictions = X_test.copy()
        test_predictions['actual_duration'] = y_test.values
        test_predictions['phone'] = native_df.loc[X_test.index, 'phone'].values
        
        models, feature_importances = train_optimized_models(
            X_train, y_train, X_test, y_test, test_predictions
        )
        
        # 4. Results Processing
        print("\n[4/4] Analyzing and saving results...")
        
        # Analyze results
        metrics_df, results = analyze_results(test_predictions, models, feature_importances)
        
        # Save outputs
        OutputHandler.write_dataframe(test_predictions, TEST_LEVEL_OUTPUT_CSV)
        OutputHandler.write_dataframe(metrics_df, OVERALL_METRICS_CSV)
        
        print("\nModel Performance:")
        print(metrics_df.to_string(index=False))
        
        # Save models
        for name, model in models.items():
            if 'keras' in str(type(model)):
                model.save(os.path.join(ERROR_ANALYSIS_DIR, f'{name.lower()}_model.h5'))
            else:
                import joblib
                joblib.dump(model, os.path.join(ERROR_ANALYSIS_DIR, f'{name.lower()}_model.pkl'))
        
        print("\nPipeline completed successfully")

    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()