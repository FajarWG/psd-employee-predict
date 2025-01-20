# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocessing import preprocess_data

def train_model(train_path, test_path):
    x_resample, y_resample, test, label_encoders, scaler = preprocess_data(train_path, test_path)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size=0.2, random_state=0)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save the model and preprocessing objects
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Training Accuracy:", model.score(X_train, y_train))
    print("Testing Accuracy:", model.score(X_test, y_test))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    print("Model and preprocessing objects saved successfully.")

    return {
        'training_accuracy': model.score(X_train, y_train),
        'testing_accuracy': model.score(X_test, y_test),
        'classification_report': classification_report(y_test, y_pred),
    }
