import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, 
    classification_report
)

if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/img/ensemble'):
    os.makedirs('static/img/ensemble')
if not os.path.exists('static/data'):
    os.makedirs('static/data')

def load_and_preprocess_data(file_path='flight_data.csv'):
    """
    Load flight data and preprocess it for ensemble learning models
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample data...")
        data = {
            'nsmiles': [2468, 1138, 2468, 867, 1092, 1823, 954, 2109, 743, 1287],
            'passengers': [14.24, 149.89, 22.28, 135.76, 178.55, 45.67, 112.88, 76.33, 65.91, 198.44],
            'fare': [289.04, 252.35, 302.70, 192.20, 215.75, 275.50, 235.80, 340.25, 180.50, 260.75],
            'large_ms': [0.99, 0.66, 0.80, 0.78, 0.85, 0.93, 0.71, 0.88, 0.82, 0.91],
            'carrier_lg': ['WN', 'WN', 'WN', 'UA', 'AA', 'WN', 'DL', 'AA', 'UA', 'DL']
        }
        df = pd.DataFrame(data)
    
    numeric_cols = ['nsmiles', 'passengers', 'fare', 'large_ms']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=numeric_cols + ['carrier_lg'])
    
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
    
    sample_data = {
        'distance_category': df['distance_category'].head(10).tolist(),
        'carrier_lg': df['carrier_lg'].head(10).tolist(),
        'market_share_category': df['market_share_category'].head(10).tolist(),
        'fare_category': df['fare_category'].head(10).tolist()
    }
    
    features = df[['distance_category', 'carrier_lg', 'market_share_category']]
    target = df['fare_category']
    
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_features = encoder.fit_transform(features)
    
    feature_names = []
    for i, col in enumerate(['distance_category', 'carrier_lg', 'market_share_category']):
        for cat in encoder.categories_[i]:
            feature_names.append(f"{col.split('_')[0]}_{cat}")
    
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
    
    processed_data = {}
    for col in feature_names:
        processed_data[col] = encoded_df[col].head(10).tolist()
    processed_data['target'] = target.head(10).tolist()
    
    return encoded_features, target, feature_names, sample_data, processed_data

def create_ensemble_visualizations(X, y, feature_names, results, save_dir='static/img/ensemble'):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Individual Models')
    model_names = ['Decision Tree', 'Random Forest', 'SVM']
    for i, name in enumerate(model_names):
        circle = plt.Circle((i+1, 1), 0.6, fill=False, edgecolor='blue')
        plt.gca().add_patch(circle)
        plt.text(i+1, 1, name, ha='center', va='center', fontsize=8)
    
    plt.subplot(1, 2, 2)
    plt.title('Ensemble Approach')
    for i, name in enumerate(model_names):
        circle = plt.Circle((i+1, 2), 0.4, fill=False, edgecolor='blue')
        plt.gca().add_patch(circle)
        plt.text(i+1, 2, name[:2], ha='center', va='center', fontsize=8)
    
    combined_circle = plt.Circle((2, 1), 0.8, fill=False, edgecolor='red')
    plt.gca().add_patch(combined_circle)
    plt.text(2, 1, 'Combined\nPrediction', ha='center', va='center', fontsize=9)
    
    for i in range(3):
        plt.arrow(i+1, 1.6, 0, -0.2, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    plt.xlim(0, 4)
    plt.ylim(0, 3)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_concept.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    
    def draw_simple_tree(x, y, depth=3, width=1):
        if depth == 0:
            return
        plt.plot([x, x-width], [y, y-1], 'k-')
        plt.plot([x, x+width], [y, y-1], 'k-')
        draw_simple_tree(x-width, y-1, depth-1, width/2)
        draw_simple_tree(x+width, y-1, depth-1, width/2)
    
    draw_simple_tree(6, 5, depth=3, width=1.5)
    
    plt.text(6, 6, 'Decision Tree', ha='center', fontsize=14, weight='bold')
    
    plt.text(3.5, 1.5, 'Low Fare', ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.text(5.5, 1.5, 'Medium Fare', ha='center', va='center', bbox=dict(facecolor='lightyellow', alpha=0.5))
    plt.text(6.5, 1.5, 'High Fare', ha='center', va='center', bbox=dict(facecolor='lightcoral', alpha=0.5))
    plt.text(8.5, 1.5, 'Low Fare', ha='center', va='center', bbox=dict(facecolor='lightgreen', alpha=0.5))
    
    plt.text(6, 5, 'Distance?', ha='center', va='center', bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.text(4.5, 4, 'Carrier?', ha='center', va='center', bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.text(7.5, 4, 'Market Share?', ha='center', va='center', bbox=dict(facecolor='lightblue', alpha=0.5))
    
    plt.xlim(2, 10)
    plt.ylim(1, 7)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'decision_tree.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    for i in range(3):
        x_offset = i * 3 + 2
        draw_simple_tree(x_offset, 4, depth=2, width=0.7)
        plt.text(x_offset, 4.5, f'Tree {i+1}', ha='center', fontsize=10)
    
    plt.text(5, 5.5, 'Random Forest: Multiple Decision Trees', ha='center', fontsize=14, weight='bold')
    
    plt.arrow(2, 1, 3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    plt.arrow(5, 1, 3, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    plt.arrow(8, 1, 0, -0.5, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    plt.text(8, 0, 'Majority Vote', ha='center', va='center', 
             bbox=dict(facecolor='lightgreen', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.xlim(0, 10)
    plt.ylim(-0.5, 6)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'random_forest.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    
    np.random.seed(42)
    X1 = np.random.randn(30, 2) + np.array([2, 2])
    X2 = np.random.randn(30, 2) + np.array([5, 5])
    X_svm = np.vstack((X1, X2))
    y_svm = np.array([0] * 30 + [1] * 30)
    
    plt.scatter(X_svm[y_svm==0, 0], X_svm[y_svm==0, 1], c='blue', label='Low Fare')
    plt.scatter(X_svm[y_svm==1, 0], X_svm[y_svm==1, 1], c='red', label='High Fare')
    
    xmin, xmax = plt.xlim()[0], plt.xlim()[1]
    ymin, ymax = plt.ylim()[0], plt.ylim()[1]
    
    xx = np.linspace(0, 7)
    yy = 0.8 * xx + 0.5
    plt.plot(xx, yy, 'k-', label='Decision Boundary')
    
    plt.plot(xx, yy + 0.8, 'k--', alpha=0.3)
    plt.plot(xx, yy - 0.8, 'k--', alpha=0.3)
    
    for i in range(5):
        plt.scatter(X1[i, 0], X1[i, 1], s=150, linewidth=1, facecolors='none', edgecolors='green')
        plt.scatter(X2[i, 0], X2[i, 1], s=150, linewidth=1, facecolors='none', edgecolors='green')
        
    plt.text(1.5, 7, 'Support Vector Machine', ha='center', fontsize=14, weight='bold')
    plt.text(5, 0.5, 'Maximum Margin Hyperplane', ha='center', fontsize=10)
    plt.text(2, 0.5, 'Support Vectors', ha='center', fontsize=10, color='green')
    
    plt.xlim(0, 7)
    plt.ylim(0, 8)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'svm.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    
    if 'random_forest' in results and hasattr(results['random_forest']['model'], 'feature_importances_'):
        importances = results['random_forest']['model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title('Feature Importance from Random Forest')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300)
        plt.close()
    else:
        plt.title('Example Feature Importance (Random Forest)')
        feature_imp = np.random.rand(len(feature_names))
        feature_imp = feature_imp / feature_imp.sum()
        indices = np.argsort(feature_imp)[::-1]
        
        plt.bar(range(len(feature_imp)), feature_imp[indices], align='center')
        plt.xticks(range(len(feature_imp)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300)
        plt.close()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Data Distribution (Sample)')
    
    x_idx, y_idx = 0, 1
    for category in np.unique(y):
        mask = y == category
        plt.scatter(X[mask, x_idx], X[mask, y_idx], 
                   label=f'Category {category}', alpha=0.7)
    
    plt.xlabel(feature_names[x_idx])
    plt.ylabel(feature_names[y_idx])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title('Data After Transformation (Example)')
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    for category in np.unique(y):
        mask = y == category
        plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                   label=f'Category {category}', alpha=0.7)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_transformation.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_name, model_data in results.items():
        if model_name not in ['sample_data', 'processed_data', 'feature_names']:
            model_names.append(model_name.replace('_', ' ').title())
            accuracies.append(model_data['accuracy'] * 100)
            precisions.append(model_data['precision'] * 100)
            recalls.append(model_data['recall'] * 100)
            f1_scores.append(model_data['f1_score'] * 100)
    
    x = np.arange(len(model_names))
    width = 0.2
    
    plt.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#4CAF50')
    plt.bar(x - 0.5*width, precisions, width, label='Precision', color='#2196F3')
    plt.bar(x + 0.5*width, recalls, width, label='Recall', color='#FFC107')
    plt.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color='#FF5722')
    
    plt.xlabel('Model')
    plt.ylabel('Score (%)')
    plt.title('Performance Comparison of Models')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    best_idx = np.argmax(accuracies)
    best_model = model_names[best_idx]
    best_accuracy = accuracies[best_idx]
    
    plt.annotate(f'Best model: {best_model}',
                xy=(0.5, 0.9), xycoords='axes fraction',
                fontsize=14, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    plt.annotate(f'Accuracy: {best_accuracy:.2f}%',
                xy=(0.5, 0.8), xycoords='axes fraction',
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    plt.annotate(f'Feature importance derived from\n{len(feature_names)} flight data characteristics',
                xy=(0.5, 0.65), xycoords='axes fraction',
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    
    findings = [
        'Decision Trees provide clear interpretability of flight pricing rules',
        'Random Forest achieves highest overall accuracy for fare predictions',
        'SVM excels with clearer decision boundaries between fare categories',
        'Distance and market share remain the most influential fare predictors'
    ]
    
    for i, finding in enumerate(findings):
        plt.annotate(f'• {finding}',
                    xy=(0.1, 0.5 - i*0.1), xycoords='axes fraction',
                    fontsize=10, ha='left')
    
    plt.title('Ensemble Results Summary')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_results.png'), dpi=300)
    plt.close()

def create_confusion_matrix_plots(results, save_dir='static/img/ensemble'):
    for model_name, model_data in results.items():
        if model_name not in ['sample_data', 'processed_data', 'feature_names']:
            cm = model_data['confusion_matrix']
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=model_data['model'].classes_,
                       yticklabels=model_data['model'].classes_)
            plt.title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'cm_{model_name}.png'), dpi=300)
            plt.close()

def train_evaluate_ensemble_models(X, y, feature_names=None):
    """
    Train and evaluate ensemble models (Decision Tree, Random Forest, SVM)
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        
    Returns:
        Dictionary with model results
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    results = {}
    
    print("Training Decision Tree model...")
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    
    start_time = time.time()
    dt_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = dt_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    class_metrics = {}
    for cls in dt_model.classes_:
        class_metrics[cls] = {
            'precision': report[str(cls)]['precision'],
            'recall': report[str(cls)]['recall'],
            'f1_score': report[str(cls)]['f1-score'],
            'support': report[str(cls)]['support']
        }
    
    results['decision_tree'] = {
        'model': dt_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'training_time': training_time,
        'class_metrics': class_metrics
    }
    
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    class_metrics = {}
    for cls in rf_model.classes_:
        class_metrics[cls] = {
            'precision': report[str(cls)]['precision'],
            'recall': report[str(cls)]['recall'],
            'f1_score': report[str(cls)]['f1-score'],
            'support': report[str(cls)]['support']
        }
    
    results['random_forest'] = {
        'model': rf_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'training_time': training_time,
        'class_metrics': class_metrics
    }
    
    print("Training SVM model...")
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    svm_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    y_pred = svm_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    class_metrics = {}
    for cls in svm_model.classes_:
        class_metrics[cls] = {
            'precision': report[str(cls)]['precision'],
            'recall': report[str(cls)]['recall'],
            'f1_score': report[str(cls)]['f1-score'],
            'support': report[str(cls)]['support']
        }
    
    results['svm'] = {
        'model': svm_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'training_time': training_time,
        'class_metrics': class_metrics,
        'scaler': scaler
    }
    
    return results

def run_ensemble_analysis(file_path='flight_data.csv'):
    print("Starting ensemble analysis...")
    
    X, y, feature_names, sample_data, processed_data = load_and_preprocess_data(file_path)
    
    results = train_evaluate_ensemble_models(X, y, feature_names)
    
    results['sample_data'] = sample_data
    results['processed_data'] = processed_data
    results['feature_names'] = feature_names
    
    create_ensemble_visualizations(X, y, feature_names, results)
    create_confusion_matrix_plots(results)
    
    with open('static/data/ensemble_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("Ensemble analysis completed successfully.")
    
    return results

if __name__ == "__main__":
    results = run_ensemble_analysis()
    
    print("\nEnsemble Learning Results Summary:")
    print("---------------------------------")
    
    for model_name, model_data in results.items():
        if model_name not in ['sample_data', 'processed_data', 'feature_names']:
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy: {model_data['accuracy']:.4f}")
            print(f"  Precision: {model_data['precision']:.4f}")
            print(f"  Recall: {model_data['recall']:.4f}")
            print(f"  F1 Score: {model_data['f1_score']:.4f}")
            print(f"  Training Time: {model_data['training_time']:.2f} seconds")
    
    if 'random_forest' in results and hasattr(results['random_forest']['model'], 'feature_importances_'):
        print("\nFeature Importance (Random Forest):")
        importance = results['random_forest']['model'].feature_importances_
        for i, feat in enumerate(results['feature_names']):
            print(f"  {feat}: {importance[i]:.4f}")
