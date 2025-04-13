"""
Contains enhanced modeling approaches with advanced feature engineering and models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib

class FeatureEngineer:
    """Enhanced feature engineering with sequence modeling support"""
    
    @staticmethod
    def add_advanced_features(df):
        """Adds advanced linguistic and sequence features"""
        # Phone class features
        df['is_plosive'] = df['phone'].str.lower().isin(['p', 'b', 't', 'd', 'k', 'g']).astype(int)
        df['is_fricative'] = df['phone'].str.lower().isin(['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h']).astype(int)
        df['is_nasal'] = df['phone'].str.lower().isin(['m', 'n', 'ŋ']).astype(int)
        
        # Position features
        df['position_in_utterance'] = df.groupby('file').cumcount()
        df['utterance_length'] = df.groupby('file')['phone'].transform('count')
        df['position_ratio_utterance'] = df['position_in_utterance'] / df['utterance_length']
        
        # Context windows
        for window in [1, 2]:
            df[f'prev_{window}_phone'] = df.groupby('file')['phone'].shift(window)
            df[f'next_{window}_phone'] = df.groupby('file')['phone'].shift(-window)
        
        return df

    @staticmethod
    def prepare_sequence_features(df, sequence_length=5):
        """Prepares data for sequence models"""
        sequences = []
        phone_groups = df.groupby('file')
        
        for _, group in phone_groups:
            group = group.sort_values('position_in_utterance')
            for i in range(len(group) - sequence_length + 1):
                seq = group.iloc[i:i + sequence_length]
                sequences.append(seq)
                
        return pd.concat(sequences)

class DurationModeler:
    """Enhanced duration modeling with advanced techniques"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.phone_encoder = None
        self.best_model = None
        self.performance_metrics = []

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, test_predictions):
        """Enhanced training pipeline"""
        # Feature selection
        self._perform_feature_selection(X_train, y_train)
        
        # Train and compare models
        self._train_random_forest(X_train, y_train, X_test, y_test, test_predictions)
        self._train_gradient_boosting(X_train, y_train, X_test, y_test, test_predictions)
        self._train_hybrid_model(X_train, y_train, X_test, y_test, test_predictions)
        
        # Select best model
        self._select_best_model()

    def _perform_feature_selection(self, X, y):
        """Selects most important features"""
        self.feature_selector = SelectKBest(f_regression, k=15)
        X_selected = self.feature_selector.fit_transform(X, y)
        return X_selected

    def _train_random_forest(self, X_train, y_train, X_test, y_test, test_predictions):
        """Optimized Random Forest with hyperparameter tuning"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        self.models['RandomForest'] = best_rf
        test_predictions['RandomForest_pred'] = y_pred
        self._store_metrics('RandomForest', y_test, y_pred)
        
        print(f"RandomForest best params: {grid_search.best_params_}")
        print(f"Feature importances: {sorted(zip(X_train.columns, best_rf.feature_importances_), key=lambda x: x[1], reverse=True)}")

    def _train_gradient_boosting(self, X_train, y_train, X_test, y_test, test_predictions):
        """Gradient Boosting model"""
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        
        self.models['GradientBoosting'] = gb
        test_predictions['GradientBoosting_pred'] = y_pred
        self._store_metrics('GradientBoosting', y_test, y_pred)

    def _train_hybrid_model(self, X_train, y_train, X_test, y_test, test_predictions):
        """Hybrid neural network with sequence modeling"""
        # Standard scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['HybridNN'] = scaler
        
        # Neural network architecture
        input_layer = Input(shape=(X_train_scaled.shape[1],))
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        output = Dense(1)(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(0.001), loss='mae')
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=1
        )
        
        y_pred = model.predict(X_test_scaled).flatten()
        self.models['HybridNN'] = model
        test_predictions['HybridNN_pred'] = y_pred
        self._store_metrics('HybridNN', y_test, y_pred)

    def _store_metrics(self, model_name, y_true, y_pred):
        """Stores evaluation metrics"""
        metrics = {
            'Model': model_name,
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        self.performance_metrics.append(metrics)

    def _select_best_model(self):
        """Selects the best performing model"""
        metrics_df = pd.DataFrame(self.performance_metrics)
        best_model_name = metrics_df.loc[metrics_df['MAE'].idxmin(), 'Model']
        self.best_model = self.models[best_model_name]
        print(f"\nBest model selected: {best_model_name}")
        print(metrics_df)

    def save_models(self, output_dir):
        """Saves trained models and artifacts"""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models.items():
            if 'keras' in str(type(model)):
                model.save(os.path.join(output_dir, f'{name}.h5'))
            else:
                joblib.dump(model, os.path.join(output_dir, f'{name}.pkl'))
        
        if self.scalers:
            joblib.dump(self.scalers, os.path.join(output_dir, 'scalers.pkl'))
        
        print(f"Models saved to {output_dir}")

    def apply_to_non_native(self, X_non_native):
        """Applies all models to non-native data"""
        predictions = {}
        for name, model in self.models.items():
            if 'HybridNN' in name:
                scaler = self.scalers.get(name)
                X_scaled = scaler.transform(X_non_native)
                predictions[name] = model.predict(X_scaled).flatten()
            else:
                predictions[name] = model.predict(X_non_native)
        
        return pd.DataFrame(predictions)