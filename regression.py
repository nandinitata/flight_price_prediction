import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Create directories if they don't exist
if not os.path.exists('static/img'):
    os.makedirs('static/img')
if not os.path.exists('static/data'):
    os.makedirs('static/data')

def load_flight_data(filename='flight_data.csv'):
    """Load flight data from CSV file"""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_flight_data(df):
    """Preprocess flight data for regression and classification analysis"""
    # Select relevant columns
    numeric_cols = ['nsmiles', 'passengers', 'fare', 'large_ms']
    
    # Convert numeric columns to appropriate types
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=numeric_cols + ['carrier_lg'])
    
    # Create categorical features
    df['distance_category'] = pd.cut(
        df['nsmiles'],
        bins=[0, 500, 1000, 2000, float('inf')],
        labels=['Short', 'Medium', 'Long', 'VeryLong']
    )
    
    df['fare_category'] = pd.cut(
        df['fare'],
        bins=[0, 150, 250, 350, 500, float('inf')],
        labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
    )
    
    df['market_share_category'] = pd.cut(
        df['large_ms'],
        bins=[0, 0.33, 0.66, 1],
        labels=['LowShare', 'MediumShare', 'HighShare']
    )
    
    # Create binary target for logistic regression (High fare vs not high fare)
    df['high_fare'] = df['fare_category'].isin(['High', 'VeryHigh']).astype(int)
    
    return df

def create_scatter_visualization(df):
    """Create a scatter plot of distance vs fare colored by carrier"""
    plt.figure(figsize=(10, 6))
    
    # Limit to top 5 carriers for clearer visualization
    top_carriers = df['carrier_lg'].value_counts().nlargest(5).index.tolist()
    df_plot = df[df['carrier_lg'].isin(top_carriers)]
    
    sns.scatterplot(data=df_plot, x='nsmiles', y='fare', hue='carrier_lg', alpha=0.7)
    
    plt.title('Distance vs Fare by Carrier')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Fare ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/img/distance_vs_fare.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_linear_regression_visualization(X_train, X_test, y_train, y_test, lr_model):
    """Create linear regression visualization for flight fare prediction"""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Testing data')
    
    # Sort X_test for smoother line plot
    sorted_indices = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sorted_indices]
    
    # Predict on sorted X_test
    y_pred = lr_model.predict(X_test_sorted)
    
    plt.plot(X_test_sorted, y_pred, color='red', linewidth=2, label='Linear regression')
    
    plt.title('Linear Regression: Distance vs Fare')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Fare ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/img/linear_regression_flight.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_logistic_regression_visualization(X_train, X_test, y_train, y_test, log_model):
    """Create logistic regression visualization for high fare prediction"""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Testing data')
    
    # Create a range of X values for the decision boundary
    X_range = np.linspace(X_train.min() - 100, X_train.max() + 100, 300).reshape(-1, 1)
    
    # Predict probabilities for the range
    y_proba = log_model.predict_proba(X_range)[:, 1]
    
    plt.plot(X_range, y_proba, color='red', linewidth=2, label='Logistic regression (probability)')
    
    # Draw decision boundary at probability = 0.5
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision boundary (p=0.5)')
    
    plt.title('Logistic Regression: Distance vs Probability of High Fare')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Probability of High Fare')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/img/logistic_regression_flight.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sigmoid_visualization():
    """Create a visualization of the sigmoid function used in logistic regression"""
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    
    plt.xlabel('z = β₀ + β₁X')
    plt.ylabel('σ(z) = 1 / (1 + e^(-z))')
    plt.title('Sigmoid (Logistic) Function')
    
    plt.text(6, 0.2, 'σ(z) → 1 as z → ∞', fontsize=12)
    plt.text(-8, 0.8, 'σ(z) → 0 as z → -∞', fontsize=12)
    plt.text(1, 0.55, 'σ(0) = 0.5', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('static/img/sigmoid_function.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_linear_regression(df):
    """Perform linear regression to predict fare based on distance"""
    # Prepare data
    X = df[['nsmiles']].values
    y = df['fare'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save sample data for visualization
    train_sample = pd.DataFrame({
        'nsmiles': X_train.flatten()[:5],
        'fare': y_train[:5]
    })
    train_sample.to_csv('static/data/train_sample_lr.csv', index=False)
    
    test_sample = pd.DataFrame({
        'nsmiles': X_test.flatten()[:5],
        'fare': y_test[:5],
        'predicted_fare': y_pred[:5]
    })
    test_sample.to_csv('static/data/test_sample_lr.csv', index=False)
    
    # Create visualization
    create_linear_regression_visualization(X_train, X_test, y_train, y_test, lr)
    
    return {
        'model': lr,
        'mse': mse,
        'r2': r2,
        'coefficients': {
            'intercept': lr.intercept_,
            'distance': lr.coef_[0]
        },
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def run_logistic_regression(df):
    """Perform logistic regression to predict high fare (binary) based on distance"""
    # Prepare data
    X = df[['nsmiles']].values
    y = df['high_fare'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = log_reg.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Save sample data for visualization
    train_sample = pd.DataFrame({
        'nsmiles': X_train.flatten()[:5],
        'high_fare': y_train[:5]
    })
    train_sample.to_csv('static/data/train_sample_logreg.csv', index=False)
    
    test_sample = pd.DataFrame({
        'nsmiles': X_test.flatten()[:5],
        'high_fare': y_test[:5],
        'predicted_high_fare': y_pred[:5]
    })
    test_sample.to_csv('static/data/test_sample_logreg.csv', index=False)
    
    # Create visualization - note that we use the unscaled X for visualization but the model was trained on scaled X
    create_logistic_regression_visualization(X_train, X_test, y_train, y_test, 
                                            LogisticRegression(max_iter=1000).fit(X_train, y_train))
    
    # Create sigmoid function visualization
    create_sigmoid_visualization()
    
    return {
        'model': log_reg,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }

def run_decision_tree(df):
    """Perform decision tree classification to predict high fare based on distance"""
    # Prepare data
    X = df[['nsmiles']].values
    y = df['high_fare'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train decision tree model
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': dt,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def run_naive_bayes(df):
    """Perform multinomial naive bayes to predict high fare based on distance"""
    # For Multinomial NB, we need non-negative features
    # Prepare data
    X = df[['nsmiles']].values
    X_positive = X - X.min() if X.min() < 0 else X  # Ensure non-negative
    y = df['high_fare'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_positive, y, test_size=0.3, random_state=42)
    
    # Train multinomial naive bayes model
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    
    # Make predictions
    y_pred = nb.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': nb,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def create_model_comparison(log_results, dt_results, nb_results):
    """Create a comparison of logistic regression, decision tree, and naive bayes for flight fare prediction"""
    models = ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
    accuracies = [log_results['accuracy'], dt_results['accuracy'], nb_results['accuracy']]
    precisions = [log_results['precision'], dt_results['precision'], nb_results['precision']]
    recalls = [log_results['recall'], dt_results['recall'], nb_results['recall']]
    f1_scores = [log_results['f1'], dt_results['f1'], nb_results['f1']]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.2
    
    plt.bar(x - width*1.5, [acc*100 for acc in accuracies], width, label='Accuracy', color='blue')
    plt.bar(x - width/2, [prec*100 for prec in precisions], width, label='Precision', color='green')
    plt.bar(x + width/2, [rec*100 for rec in recalls], width, label='Recall', color='orange')
    plt.bar(x + width*1.5, [f1*100 for f1 in f1_scores], width, label='F1-Score', color='purple')
    
    plt.title('Model Performance Comparison for Flight Fare Prediction')
    plt.ylabel('Score (%)')
    plt.ylim(0, 100)
    plt.xticks(x, models, rotation=15)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(accuracies):
        plt.text(i - width*1.5, v*100 + 2, f'{v*100:.1f}%', ha='center', fontsize=9)
    for i, v in enumerate(precisions):
        plt.text(i - width/2, v*100 + 2, f'{v*100:.1f}%', ha='center', fontsize=9)
    for i, v in enumerate(recalls):
        plt.text(i + width/2, v*100 + 2, f'{v*100:.1f}%', ha='center', fontsize=9)
    for i, v in enumerate(f1_scores):
        plt.text(i + width*1.5, v*100 + 2, f'{v*100:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('static/img/model_comparison_flight.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save performance metrics to csv
    performance = pd.DataFrame({
        'Model': models,
        'Accuracy': [acc*100 for acc in accuracies],
        'Precision': [prec*100 for prec in precisions],
        'Recall': [rec*100 for rec in recalls],
        'F1_Score': [f1*100 for f1 in f1_scores]
    })
    performance.to_csv('static/data/model_performance_flight.csv', index=False)

def create_confusion_matrices(log_results, dt_results, nb_results):
    """Create confusion matrices visualization for all flight fare prediction models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.heatmap(log_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Logistic Regression\nAccuracy: {log_results["accuracy"]:.2f}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['Regular Fare', 'High Fare'])
    axes[0].set_yticklabels(['Regular Fare', 'High Fare'])
    
    sns.heatmap(dt_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'Decision Tree\nAccuracy: {dt_results["accuracy"]:.2f}')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticklabels(['Regular Fare', 'High Fare'])
    axes[1].set_yticklabels(['Regular Fare', 'High Fare'])
    
    sns.heatmap(nb_results['confusion_matrix'], annot=True, fmt='d', cmap='Oranges', ax=axes[2])
    axes[2].set_title(f'Naive Bayes\nAccuracy: {nb_results["accuracy"]:.2f}')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('Actual')
    axes[2].set_xticklabels(['Regular Fare', 'High Fare'])
    axes[2].set_yticklabels(['Regular Fare', 'High Fare'])
    
    plt.tight_layout()
    plt.savefig('static/img/confusion_matrices_flight.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_regression_analysis():
    """Main function to perform all regression analyses on flight data"""
    # Load data
    df = load_flight_data()
    if df is None:
        print("Error: Could not load flight data.")
        return None
    
    # Preprocess data
    df_processed = preprocess_flight_data(df)
    
    # Create initial visualization
    create_scatter_visualization(df_processed)
    
    # Run linear regression
    lr_results = run_linear_regression(df_processed)
    
    # Run logistic regression
    log_results = run_logistic_regression(df_processed)
    
    # Run decision tree
    dt_results = run_decision_tree(df_processed)
    
    # Run naive bayes
    nb_results = run_naive_bayes(df_processed)
    
    # Create model comparison
    create_model_comparison(log_results, dt_results, nb_results)
    
    # Create confusion matrices visualization
    create_confusion_matrices(log_results, dt_results, nb_results)
    
    # Return all results
    return {
        'linear_regression': lr_results,
        'logistic_regression': log_results,
        'decision_tree': dt_results,
        'naive_bayes': nb_results,
        'train_samples': {
            'linear': pd.read_csv('static/data/train_sample_lr.csv'),
            'logistic': pd.read_csv('static/data/train_sample_logreg.csv')
        },
        'test_samples': {
            'linear': pd.read_csv('static/data/test_sample_lr.csv'),
            'logistic': pd.read_csv('static/data/test_sample_logreg.csv')
        }
    }

if __name__ == "__main__":
    results = perform_regression_analysis()
    if results:
        print("Flight data regression analysis completed successfully.")
        
        # Print linear regression results
        lr = results['linear_regression']
        print(f"\nLinear Regression Results:")
        print(f"MSE: {lr['mse']:.2f}")
        print(f"R²: {lr['r2']:.2f}")
        print(f"Intercept: {lr['coefficients']['intercept']:.2f}")
        print(f"Distance Coefficient: {lr['coefficients']['distance']:.4f}")
        
        # Print logistic regression results
        log = results['logistic_regression']
        print(f"\nLogistic Regression Results:")
        print(f"Accuracy: {log['accuracy']:.2f}")
        print(f"Precision: {log['precision']:.2f}")
        print(f"Recall: {log['recall']:.2f}")
        print(f"F1 Score: {log['f1']:.2f}")
        
        # Print decision tree results
        dt = results['decision_tree']
        print(f"\nDecision Tree Results:")
        print(f"Accuracy: {dt['accuracy']:.2f}")
        
        # Print naive bayes results
        nb = results['naive_bayes']
        print(f"\nNaive Bayes Results:")
        print(f"Accuracy: {nb['accuracy']:.2f}")