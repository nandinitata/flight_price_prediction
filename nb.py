import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import OneHotEncoder

# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/img'):
    os.makedirs('static/img')

def visualize_class_prediction_probabilities(model, X_test, y_test, classes, save_path='static/img/nb_class_probabilities.png'):
    """Visualize the probability distributions for each class"""
    probs = model.predict_proba(X_test)
    
    plt.figure(figsize=(14, 8))
    
    # Select a sample of test instances for each class
    samples_per_class = 5
    plt.suptitle('Probability Distribution by Class', fontsize=16)
    
    for i, cls in enumerate(classes):
        # Get indices of samples from this class
        indices = np.where(y_test == cls)[0]
        if len(indices) > samples_per_class:
            indices = indices[:samples_per_class]
        
        # Plot probabilities for these samples
        for j, idx in enumerate(indices):
            plt.subplot(len(classes), 1, i+1)
            plt.bar(np.arange(len(classes)), probs[idx], alpha=0.7, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(classes))))
            plt.xticks(np.arange(len(classes)), classes)
            plt.ylabel(f"Class: {cls}")
            if i == 0:
                plt.title("Predicted Probabilities for Test Samples")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def load_and_preprocess_data(file_path='flight_data.csv'):
    """
    Load flight data and preprocess it for Naive Bayes classification
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

def train_evaluate_nb_models(X, y, feature_names):
    """
    Train and evaluate both Multinomial and Bernoulli Naive Bayes models
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train and time Multinomial NB
    mnb = MultinomialNB(alpha=1.0)  # alpha=1.0 is Laplace smoothing
    
    start_time = time.time()
    mnb.fit(X_train, y_train)
    mnb_training_time = time.time() - start_time
    
    mnb_predictions = mnb.predict(X_test)
    
    mnb_accuracy = accuracy_score(y_test, mnb_predictions)
    mnb_conf_matrix = confusion_matrix(y_test, mnb_predictions)
    mnb_report = classification_report(y_test, mnb_predictions, output_dict=True)
    
    # Calculate precision, recall, and f1 score at macro level
    mnb_precision, mnb_recall, mnb_f1, _ = precision_recall_fscore_support(
        y_test, mnb_predictions, average='macro'
    )
    
    # Train and time Bernoulli NB
    bnb = BernoulliNB(alpha=1.0)
    
    start_time = time.time()
    bnb.fit(X_train, y_train)
    bnb_training_time = time.time() - start_time
    
    bnb_predictions = bnb.predict(X_test)
    
    bnb_accuracy = accuracy_score(y_test, bnb_predictions)
    bnb_conf_matrix = confusion_matrix(y_test, bnb_predictions)
    bnb_report = classification_report(y_test, bnb_predictions, output_dict=True)
    
    # Calculate precision, recall, and f1 score at macro level
    bnb_precision, bnb_recall, bnb_f1, _ = precision_recall_fscore_support(
        y_test, bnb_predictions, average='macro'
    )
    
    # Create confusion matrix visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(mnb_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=mnb.classes_, yticklabels=mnb.classes_, ax=ax1)
    ax1.set_title('Multinomial NB Confusion Matrix\nAccuracy: {:.2f}%'.format(mnb_accuracy * 100))
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    sns.heatmap(bnb_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=bnb.classes_, yticklabels=bnb.classes_, ax=ax2)
    ax2.set_title('Bernoulli NB Confusion Matrix\nAccuracy: {:.2f}%'.format(bnb_accuracy * 100))
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('static/img/nb_confusion_matrices.png')
    plt.close()
    
    # Create feature importance visualization
    plt.figure(figsize=(14, 8))
    
    for i, cls in enumerate(mnb.classes_):
        top_indices = np.argsort(mnb.feature_log_prob_[i])[-5:]
        features = [feature_names[j] for j in top_indices]
        values = [mnb.feature_log_prob_[i][j] for j in top_indices]
        
        plt.subplot(1, len(mnb.classes_), i+1)
        plt.barh(features, values, color=plt.cm.viridis(i/len(mnb.classes_)))
        plt.title(f'Class: {cls}')
        plt.tight_layout()
    
    plt.savefig('static/img/mnb_feature_importance.png')
    plt.close()
    
    # Create class performance visualization
    plt.figure(figsize=(12, 6))
    
    classes = list(mnb_report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    mnb_f1 = [mnb_report[cls]['f1-score'] for cls in classes]
    bnb_f1 = [bnb_report[cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, mnb_f1, width, label='Multinomial NB')
    plt.bar(x + width/2, bnb_f1, width, label='Bernoulli NB')
    
    plt.xlabel('Fare Category')
    plt.ylabel('F1 Score')
    plt.title('F1 Score by Fare Category and Model Type')
    plt.xticks(x, classes)
    plt.legend()
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('static/img/nb_class_performance.png')
    plt.close()
    
    # Create class probability visualization
    visualize_class_prediction_probabilities(bnb, X_test, y_test, bnb.classes_)
    
    return {
        'multinomial': {
            'accuracy': mnb_accuracy,
            'precision': mnb_precision,
            'recall': mnb_recall,
            'f1_score': mnb_f1,
            'confusion_matrix': mnb_conf_matrix,
            'report': mnb_report,
            'model': mnb,
            'training_time': mnb_training_time
        },
        'bernoulli': {
            'accuracy': bnb_accuracy,
            'precision': bnb_precision,
            'recall': bnb_recall,
            'f1_score': bnb_f1,
            'confusion_matrix': bnb_conf_matrix,
            'report': bnb_report,
            'model': bnb,
            'training_time': bnb_training_time
        }
    }

def analyze_feature_importance(models, feature_names):
    """
    Analyze which features are most important for each class in the Naive Bayes models
    """
    results = {}
    
    for model_name, model_data in models.items():
        # Skip entries that don't have a 'model' key (like 'sample_data', 'processed_data', etc.)
        if model_name not in ['multinomial', 'bernoulli']:
            continue
            
        model = model_data['model']
        
        class_importance = {}
        for i, cls in enumerate(model.classes_):
            if model_name == 'multinomial':
                feature_importance = model.feature_log_prob_[i]
            else:
                feature_importance = model.feature_log_prob_[i] - np.mean(model.feature_log_prob_, axis=0)
            
            top_indices = np.argsort(feature_importance)[-5:]
            class_importance[cls] = {
                'features': [feature_names[j] for j in top_indices],
                'values': [feature_importance[j] for j in top_indices]
            }
        
        results[model_name] = class_importance
    
    return results

def run_naive_bayes_analysis(file_path='flight_data.csv'):
    """Run the full Naive Bayes analysis pipeline and return results"""
    X, y, feature_names, categories, sample_data, processed_data = load_and_preprocess_data(file_path)
    
    model_results = train_evaluate_nb_models(X, y, feature_names)
    model_results['sample_data'] = sample_data
    model_results['processed_data'] = processed_data
    model_results['feature_names'] = feature_names
    
    importance_results = analyze_feature_importance(model_results, feature_names)
    
    return {
        'model_results': model_results,
        'importance_results': importance_results
    }

if __name__ == "__main__":
    results = run_naive_bayes_analysis()
    importance_results = results['importance_results']
    model_results = results['model_results']

    print(model_results)
    
    for model_name, model_data in model_results.items():
        if model_name not in ['sample_data', 'processed_data', 'feature_names']:
            print(f"\n{model_name.capitalize()} Naive Bayes Results:")
            print(f"Accuracy: {model_data['accuracy']:.4f}")
            print(f"Precision: {model_data['precision']:.4f}")
            print(f"Recall: {model_data['recall']:.4f}")
            print(f"F1 Score: {', '.join(f'{s:.4f}' for s in model_data['f1_score'])}")
            print(f"Training Time: {model_data['training_time']:.4f} seconds")
            
            print("\nConfusion Matrix:")
            print(model_data['confusion_matrix'])
            
            print("\nMost Important Features by Class:")
            for cls, importance in importance_results[model_name].items():
                print(f"\nClass: {cls}")
                for feature, value in zip(importance['features'], importance['values']):
                    print(f"  {feature}: {value:.4f}")

    import pickle
    with open('static/data/model_results.pkl', 'wb') as f:
        pickle.dump(model_results, f)
    with open('static/data/importance_results.pkl', 'wb') as f:
        pickle.dump(importance_results, f)
    print("\nModel results and importance results saved as pickle files.")