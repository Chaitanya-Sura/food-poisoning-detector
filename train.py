import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_collector import DataCollector
from src.models.model_loader import train_models
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.utils.helpers import evaluate_model, plot_feature_importance, generate_report
import matplotlib.pyplot as plt

def main():
    print("Starting Food Poisoning Prediction Model Training...")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Collect and prepare data
    print("Collecting and preparing data...")
    data_collector = DataCollector()
    X, y = data_collector.prepare_training_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.joblib')
    
    # Train models
    print("Training models...")
    models = train_models(X_train_processed, y_train)
    
    # Evaluate models
    print("Evaluating models...")
    results = {}
    for name, model in models.items():
        if name == 'Deep Learning':
            y_pred = (model.predict(X_test_processed) > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test_processed)
        
        metrics = evaluate_model(y_test, y_pred)
        results[name] = metrics
        
        # Generate and save feature importance plot for tree-based models
        if name != 'Deep Learning':
            plt.figure()
            plot_feature_importance(model, X_train.columns)
            plt.savefig(f'reports/{name.lower().replace(" ", "_")}_feature_importance.png')
            plt.close()
        
        # Generate and save report
        report = generate_report(name, metrics)
        with open(f'reports/{name.lower().replace(" ", "_")}_report.txt', 'w') as f:
            f.write(report)
    
    # Save results summary
    summary = pd.DataFrame(results).T
    summary.to_csv('reports/model_comparison.csv')
    
    print("\nTraining completed! Results saved in the 'reports' directory.")
    print("\nModel Performance Summary:")
    print(summary)

if __name__ == "__main__":
    main() 