import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

# Create static directories if they don't exist
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/img'):
    os.makedirs('static/img')
if not os.path.exists('static/img/ensemble'):
    os.makedirs('static/img/ensemble')
if not os.path.exists('static/data'):
    os.makedirs('static/data')

def load_and_preprocess_data(file_path='flight_data.csv'):
    """
    Load flight data and preprocess it for ensemble learning classification
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Create sample data if file not found
        data = {
            'nsmiles': [2468, 1138, 2468, 867, 1092],
            'passengers': [14.24, 149.89, 22.28, 135.76, 178.55],
            'fare': [289.04, 252.35, 302.70, 192.20, 215.75],
            'large_ms': [0.99, 0.66, 0.80, 0.78, 0.85],
            'carrier_lg': ['WN', 'WN', 'WN', 'UA', 'AA']
        }
        df = pd.DataFrame(data)
    
    # Convert numeric columns and dropna
    numeric_cols = ['nsmiles', 'passengers', 'fare', 'large_ms']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    
    # Store sample data for the template
    sample_data = {
        'distance_category': df['distance_category'].head(10).tolist(),
        'carrier_lg': df['carrier_lg'].head(10).tolist(),
        'market_share_category': df['market_share_category'].head(10).tolist(),
        'fare_category': df['fare_category'].head(10).tolist()
    }
    
    features = df[['distance_category', 'carrier_lg', 'market_share_category']]
    target = df['fare_category']
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_features = encoder.fit_transform(features)
    
    feature_names = []
    for i, col in enumerate(['distance_category', 'carrier_lg', 'market_share_category']):
        for cat in encoder.categories_[i]:
            feature_names.append(f"{col}_{cat}")
    
    # Create a dataframe for visualization purposes
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
    
    # Store processed data for the template
    processed_data = {}
    for col in feature_names:
        processed_data[col] = encoded_df[col].head(10).tolist()
    processed_data['target'] = target.head(10).tolist()
    
    return encoded_features, target, feature_names, encoder.categories_, sample_data, processed_data

def create_ensemble_visualization():
    """Create a visual explanation of ensemble learning"""
    plt.figure(figsize=(12, 8))
    
    # Create a conceptual diagram of ensemble learning
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define positions for the models
    base_y = 0.8
    ensemble_y = 0.3
    
    # Draw individual models
    models = ['Decision Tree 1', 'Decision Tree 2', 'Decision Tree 3', 'Random Forest', 'Gradient Boosting']
    model_positions = np.linspace(0.1, 0.9, len(models))
    
    for i, (model, x_pos) in enumerate(zip(models, model_positions)):
        if i < 3:  # Base learners
            rect = plt.Rectangle((x_pos - 0.06, base_y - 0.05), 0.12, 0.1, 
                               fill=True, color='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos, base_y, model, ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Connect to ensemble
            ax.arrow(x_pos, base_y - 0.05, 0, -0.25, head_width=0.02, 
                    head_length=0.03, fc='black', ec='black')
        else:  # Ensemble methods
            rect = plt.Rectangle((x_pos - 0.06, base_y - 0.05), 0.12, 0.1, 
                               fill=True, color='lightgreen', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos, base_y, model, ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Connect to final prediction
            ax.arrow(x_pos, base_y - 0.05, 0, -0.25, head_width=0.02, 
                    head_length=0.03, fc='black', ec='black')
    
    # Draw ensemble layer
    ensemble_rect = plt.Rectangle((0.15, ensemble_y - 0.05), 0.7, 0.1, 
                                fill=True, color='yellow', edgecolor='black', linewidth=2)
    ax.add_patch(ensemble_rect)
    ax.text(0.5, ensemble_y, 'Ensemble Learning\n(Combining Multiple Models)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw final prediction
    final_rect = plt.Rectangle((0.35, 0.1), 0.3, 0.08, 
                             fill=True, color='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(final_rect)
    ax.text(0.5, 0.14, 'Final Prediction', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow from ensemble to final prediction
    ax.arrow(0.5, ensemble_y - 0.05, 0, -0.08, head_width=0.02, 
            head_length=0.03, fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Ensemble Learning: Combining Multiple Models for Better Predictions', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('static/img/ensemble/ensemble_concept.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_evaluate_ensemble_models(X, y, feature_names):
    """
    Train and evaluate various ensemble learning models
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Voting Classifier': VotingClassifier(
            estimators=[
                ('dt', DecisionTreeClassifier(random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
            ],
            voting='soft'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'report': report,
            'model': model,
            'training_time': training_time
        }
    
    # Create performance comparison visualization
    plt.figure(figsize=(12, 8))
    
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] * 100 for model in model_names]
    training_times = [results[model]['training_time'] for model in model_names]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Training time comparison
    bars2 = ax2.bar(model_names, training_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('static/img/ensemble/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrices visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=model.classes_, yticklabels=model.classes_, ax=axes[idx])
        axes[idx].set_title(f'{name} Confusion Matrix', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('static/img/ensemble/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create feature importance visualization for Random Forest
    rf_model = results['Random Forest']['model']
    feature_importance = rf_model.feature_importances_
    
    plt.figure(figsize=(12, 8))
    sorted_idx = np.argsort(feature_importance)[-15:]  # Top 15 features
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance (Top 15 Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/img/ensemble/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def run_ensemble_analysis(file_path='flight_data.csv'):
    """Run the full ensemble learning analysis pipeline"""
    # Create ensemble visualization diagram
    create_ensemble_visualization()
    
    # Load and preprocess data
    X, y, feature_names, categories, sample_data, processed_data = load_and_preprocess_data(file_path)
    
    # Train and evaluate models
    results = train_evaluate_ensemble_models(X, y, feature_names)
    
    # Add sample data to results
    results['sample_data'] = sample_data
    results['processed_data'] = processed_data
    results['feature_names'] = feature_names
    
    return results

if __name__ == "__main__":
    results = run_ensemble_analysis()
    
    print("\nEnsemble Learning Results:")
    print("=" * 50)
    
    for model_name, model_data in results.items():
        if model_name not in ['sample_data', 'processed_data', 'feature_names']:
            print(f"\n{model_name}:")
            print(f"Accuracy: {model_data['accuracy']:.4f}")
            print(f"Training Time: {model_data['training_time']:.4f} seconds")
            print(f"Classification Report:")
            print(classification_report(
                model_data['confusion_matrix'].sum(axis=1), 
                model_data['confusion_matrix'].sum(axis=0)
            ))
            
    # Save results
    with open('static/data/ensemble_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to static/data/ensemble_results.pkl")