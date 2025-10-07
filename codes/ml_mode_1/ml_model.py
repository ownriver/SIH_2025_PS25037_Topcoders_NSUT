import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime
from data_generator import generate_features_for_ml

class SedimentClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'grain_size_class'
        self.model_path = 'sediment_model.pkl'
        self.scaler_path = 'scaler.pkl'
        self.encoders_path = 'label_encoders.pkl'
        
        # Fixed categorical mappings
        self.beach_type_mapping = {'berm': 0, 'dune': 1, 'intertidal': 2}
        self.region_mapping = {'Goa': 0, 'Chennai': 1, 'Vizag': 2, 'Gujarat': 3, 'Kerala': 4, 'Odisha': 5, 'Andaman': 6}
        self.season_mapping = {'winter': 0, 'summer': 1, 'monsoon': 2, 'post_monsoon': 3}
        
        # Training-time constants for consistent feature engineering
        self.min_longitude = None
        
        # Load existing model if available
        self.load_model()
    
    def prepare_features(self, df, is_training=False):
        """Prepare features for model training or prediction"""
        df_processed = df.copy()
        
        # Validate required columns
        required_cols = ['latitude', 'longitude', 'grain_size', 'beach_type', 'region', 'timestamp']
        for col in required_cols:
            if col not in df_processed.columns:
                raise ValueError(f"Required column '{col}' missing from input data")
        
        # Encode beach type (with fallback for unknown)
        df_processed['beach_type_encoded'] = df_processed['beach_type'].map(self.beach_type_mapping).fillna(0).astype(int)
        
        # Encode region (with fallback for unknown)
        df_processed['region_encoded'] = df_processed['region'].map(self.region_mapping).fillna(7).astype(int)  # 7 = 'unknown'
        
        # Add derived features
        df_processed['distance_from_equator'] = abs(df_processed['latitude'])
        
        # Coastal position using fixed reference
        if is_training:
            self.min_longitude = df_processed['longitude'].min()
        if self.min_longitude is not None:
            df_processed['coastal_position'] = df_processed['longitude'] - self.min_longitude
        else:
            # Fallback to 0 if min_longitude not set (shouldn't happen after training)
            df_processed['coastal_position'] = 0
        
        # Add seasonal features
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
        df_processed['month'] = df_processed['timestamp'].dt.month.fillna(6).astype(int)  # Default to June
        df_processed['season'] = df_processed['month'].apply(lambda x: 
            'winter' if x in [12, 1, 2] else
            'summer' if x in [3, 4, 5] else
            'monsoon' if x in [6, 7, 8, 9] else
            'post_monsoon'
        )
        df_processed['season_encoded'] = df_processed['season'].map(self.season_mapping).fillna(2).astype(int)  # Default to monsoon
        
        # Select feature columns
        feature_columns = [
            'latitude', 'longitude', 'grain_size',
            'beach_type_encoded', 'region_encoded',
            'distance_from_equator', 'coastal_position',
            'month', 'season_encoded'
        ]
        
        return df_processed[feature_columns]
    
    def train_model(self, dataset):
        """Train the sediment classification model"""
        try:
            # Prepare features with training=True to compute constants
            X = self.prepare_features(dataset, is_training=True)
            y = dataset[self.target_column]
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train RandomForest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Save model
            self.save_model()
            
            return accuracy, report
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def predict(self, features_dict):
        """Make prediction for a single sample"""
        if self.model is None:
            raise Exception("Model not trained or loaded")
        
        try:
            # Create DataFrame from features
            df = pd.DataFrame([features_dict])
            
            # Prepare features (inference mode)
            X = self.prepare_features(df, is_training=False)
            
            # Ensure feature columns match training
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            X = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get confidence (highest probability)
            confidence = float(probabilities.max())
            
            return {
                'grain_size_class': prediction,
                'confidence': confidence,
                'probabilities': {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.model.classes_, probabilities)
                }
            }
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
    
    def predict_batch(self, dataset):
        """Make predictions for multiple samples"""
        if self.model is None:
            raise Exception("Model not trained or loaded")
        
        try:
            # Prepare features (inference mode)
            X = self.prepare_features(dataset, is_training=False)
            X = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'grain_size_class': pred,
                    'confidence': float(probs.max()),
                    'probabilities': {
                        class_name: float(prob) 
                        for class_name, prob in zip(self.model.classes_, probs)
                    }
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error making batch predictions: {str(e)}")
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        try:
            if self.model is not None:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                joblib.dump({
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'classes': list(self.model.classes_) if hasattr(self.model, 'classes_') else [],
                    'beach_type_mapping': self.beach_type_mapping,
                    'region_mapping': self.region_mapping,
                    'season_mapping': self.season_mapping,
                    'min_longitude': self.min_longitude
                }, self.encoders_path)
                return True
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
        return False
    
    def load_model(self):
        """Load existing trained model"""
        try:
            if (os.path.exists(self.model_path) and 
                os.path.exists(self.scaler_path) and 
                os.path.exists(self.encoders_path)):
                
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                
                metadata = joblib.load(self.encoders_path)
                self.feature_columns = metadata.get('feature_columns', [])
                self.target_column = metadata.get('target_column', 'grain_size_class')
                # Load categorical mappings if available
                self.beach_type_mapping = metadata.get('beach_type_mapping', self.beach_type_mapping)
                self.region_mapping = metadata.get('region_mapping', self.region_mapping)
                self.season_mapping = metadata.get('season_mapping', self.season_mapping)
                self.min_longitude = metadata.get('min_longitude')
                
                return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        return False
    
    def get_model_info(self):
        """Get information about the trained model"""
        if self.model is None:
            return None
        
        info = {
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.model.n_estimators,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'classes': list(self.model.classes_) if hasattr(self.model, 'classes_') else [],
            'feature_importance': {}
        }
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            # Sort by importance
            info['feature_importance'] = dict(
                sorted(importance_dict.items(), key=lambda x: float(x[1]), reverse=True)
            )
        
        return info

if __name__ == "__main__":
    # Test the classifier
    from data_generator import generate_synthetic_dataset
    
    # Generate test dataset
    dataset = generate_synthetic_dataset(1000)
    
    # Initialize classifier
    classifier = SedimentClassifier()
    
    # Train model
    accuracy, report = classifier.train_model(dataset)
    print(f"Model trained with accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(report)
    
    # Test prediction
    test_sample = {
        'latitude': 15.5,
        'longitude': 73.9,
        'grain_size': 0.8,
        'beach_type': 'berm',
        'region': 'Goa',
        'timestamp': '2024-01-15T10:30:00'
    }
    
    result = classifier.predict(test_sample)
    print(f"\nTest prediction: {result}")
