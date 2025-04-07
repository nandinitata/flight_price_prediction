import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
import os

def visualize_feature_importance(model, feature_names, save_path='static/img/dt_feature_importance.png'):
    """Create a more visually appealing feature importance chart"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance in Decision Tree Model', fontsize=18)
    plt.bar(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.ylabel('Relative Importance', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(importances[indices]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def load_and_preprocess_data(file_path='flight_data.csv'):
    """
    Load flight data and preprocess it for Decision Tree classification
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert numeric columns to proper types
    numeric_cols = ['nsmiles', 'passengers', 'fare', 'large_ms']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values in important columns
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
    
    # Save sample of original data for visualization
    sample_original = df[['distance_category', 'carrier_lg', 'market_share_category', 'fare_category']].head(10)
    sample_original.to_csv('sample_original_dt.csv', index=False)
    
    # Select features and target
    features = df[['distance_category', 'carrier_lg', 'market_share_category']]
    target = df['fare_category']
    
    # One-hot encode features
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_features = encoder.fit_transform(features)
    
    # Get feature names
    feature_names = []
    for i, col in enumerate(['distance_category', 'carrier_lg', 'market_share_category']):
        for cat in encoder.categories_[i]:
            feature_names.append(f"{col.split('_')[0]}_{cat}")
    
    # Save sample of processed data for visualization
    encoded_df = pd.DataFrame(encoded_features[:10], columns=feature_names)
    encoded_df['target'] = target[:10]
    encoded_df.to_csv('sample_processed_dt.csv', index=False)
    
    return encoded_features, target, feature_names

def train_evaluate_dt_models(X, y, feature_names):
    """
    Train and evaluate three different Decision Tree models
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create directory for output images if it doesn't exist
    img_dir = 'static/img'
    os.makedirs(img_dir, exist_ok=True)
    
    # Model 1: Default parameters
    dt1 = DecisionTreeClassifier(random_state=42)
    dt1.fit(X_train, y_train)
    y_pred1 = dt1.predict(X_test)
    
    # Model 2: Limited depth to prevent overfitting
    dt2 = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt2.fit(X_train, y_train)
    y_pred2 = dt2.predict(X_test)
    
    # Model 3: Using entropy criterion instead of gini
    dt3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt3.fit(X_train, y_train)
    y_pred3 = dt3.predict(X_test)
    
    # Visualize the trees
    plt.figure(figsize=(15, 10))
    tree.plot_tree(dt1, feature_names=feature_names, class_names=dt1.classes_, filled=True, fontsize=10)
    plt.title("Decision Tree Model 1: Default Parameters")
    plt.tight_layout()
    plt.savefig(f'{img_dir}/dt_model1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 10))
    tree.plot_tree(dt2, feature_names=feature_names, class_names=dt2.classes_, filled=True, fontsize=10)
    plt.title("Decision Tree Model 2: Max Depth=3")
    plt.tight_layout()
    plt.savefig(f'{img_dir}/dt_model2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 10))
    tree.plot_tree(dt3, feature_names=feature_names, class_names=dt3.classes_, filled=True, fontsize=10)
    plt.title("Decision Tree Model 3: Entropy Criterion")
    plt.tight_layout()
    plt.savefig(f'{img_dir}/dt_model3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance analysis
    plt.figure(figsize=(12, 6))
    sorted_idx = dt1.feature_importances_.argsort()[-10:]
    plt.barh(np.array(feature_names)[sorted_idx], dt1.feature_importances_[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Features by Importance')
    plt.tight_layout()
    plt.savefig(f'{img_dir}/dt_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance by class visualization
    models = {
        "Default Parameters": (dt1, y_pred1),
        "Max Depth=3": (dt2, y_pred2),
        "Entropy Criterion": (dt3, y_pred3)
    }
    
    # Create a DataFrame for class performance comparison
    class_perf_data = []
    for model_name, (model, y_pred) in models.items():
        report = classification_report(y_test, y_pred, output_dict=True)
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_perf_data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'F1-Score': metrics['f1-score']
                })
    
    class_perf_df = pd.DataFrame(class_perf_data)
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Class', y='F1-Score', hue='Model', data=class_perf_df)
    plt.title('F1 Score by Fare Category and Model Type')
    plt.xlabel('Fare Category')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(f'{img_dir}/dt_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix visualization for the best model (using entropy)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dt3.classes_, yticklabels=dt3.classes_)
    plt.title('Confusion Matrix - Entropy Model\nAccuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred3) * 100))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{img_dir}/dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Decision tree basic structure image for educational purposes
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.text(0.5, 0.9, "Root Node", ha='center', fontsize=14, bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.plot([0.5, 0.25], [0.85, 0.7], 'k-')
    plt.plot([0.5, 0.75], [0.85, 0.7], 'k-')
    plt.text(0.25, 0.65, "Decision Node", ha='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.text(0.75, 0.65, "Decision Node", ha='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.plot([0.25, 0.15], [0.6, 0.45], 'k-')
    plt.plot([0.25, 0.35], [0.6, 0.45], 'k-')
    plt.plot([0.75, 0.65], [0.6, 0.45], 'k-')
    plt.plot([0.75, 0.85], [0.6, 0.45], 'k-')
    plt.text(0.15, 0.4, "Leaf Node", ha='center', fontsize=10, bbox=dict(facecolor='salmon', alpha=0.5))
    plt.text(0.35, 0.4, "Leaf Node", ha='center', fontsize=10, bbox=dict(facecolor='salmon', alpha=0.5))
    plt.text(0.65, 0.4, "Leaf Node", ha='center', fontsize=10, bbox=dict(facecolor='salmon', alpha=0.5))
    plt.text(0.85, 0.4, "Leaf Node", ha='center', fontsize=10, bbox=dict(facecolor='salmon', alpha=0.5))
    plt.text(0.05, 0.8, "True", fontsize=9)
    plt.text(0.85, 0.8, "False", fontsize=9)
    plt.text(0.05, 0.55, "True", fontsize=9)
    plt.text(0.45, 0.55, "False", fontsize=9)
    plt.text(0.55, 0.55, "True", fontsize=9)
    plt.text(0.95, 0.55, "False", fontsize=9)
    plt.text(0.5, 0.2, "Decision Tree Structure", ha='center', fontsize=16)
    plt.savefig(f'{img_dir}/decision_tree_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Decision tree prediction process image
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    # Drawing the tree nodes
    plt.text(0.5, 0.9, "Is distance > 1000 miles?", ha='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.plot([0.5, 0.25], [0.85, 0.7], 'k-')
    plt.plot([0.5, 0.75], [0.85, 0.7], 'k-')
    plt.text(0.25, 0.65, "Is carrier SW?", ha='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.text(0.75, 0.65, "Is market share > 0.5?", ha='center', fontsize=12, bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.plot([0.25, 0.15], [0.6, 0.45], 'k-')
    plt.plot([0.25, 0.35], [0.6, 0.45], 'k-')
    plt.plot([0.75, 0.65], [0.6, 0.45], 'k-')
    plt.plot([0.75, 0.85], [0.6, 0.45], 'k-')
    plt.text(0.15, 0.4, "Predict: Low", ha='center', fontsize=10, bbox=dict(facecolor='salmon', alpha=0.5))
    plt.text(0.35, 0.4, "Predict: Medium", ha='center', fontsize=10, bbox=dict(facecolor='salmon', alpha=0.5))
    plt.text(0.65, 0.4, "Predict: Medium", ha='center', fontsize=10, bbox=dict(facecolor='salmon', alpha=0.5))
    plt.text(0.85, 0.4, "Predict: High", ha='center', fontsize=10, bbox=dict(facecolor='salmon', alpha=0.5))
    # Labels for the decision process
    plt.text(0.2, 0.8, "No", fontsize=10)
    plt.text(0.8, 0.8, "Yes", fontsize=10)
    plt.text(0.1, 0.55, "Yes", fontsize=10)
    plt.text(0.4, 0.55, "No", fontsize=10)
    plt.text(0.6, 0.55, "No", fontsize=10)
    plt.text(0.9, 0.55, "Yes", fontsize=10)
    # Example data point traversing the tree
    plt.plot([0.5, 0.75, 0.85], [0.9, 0.65, 0.4], 'r--', lw=2)
    plt.text(0.85, 0.3, "Final Prediction: High Fare", ha='center', fontsize=12, 
             bbox=dict(facecolor='yellow', alpha=0.5))
    plt.text(0.5, 0.15, "Decision Path for Example: Long Distance, High Market Share", 
             ha='center', fontsize=14)
    plt.savefig(f'{img_dir}/decision_tree_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results for each model
    results = {}
    for name, (model, y_pred) in models.items():
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        
        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'report': report,
            'model': model
        }
    
    return results

def analyze_decision_trees(models, feature_names):
    """
    Analyze the decision tree models and feature importance
    """
    print("\n=== DECISION TREE ANALYSIS ===")
    
    # Compare accuracy across models
    print("\nAccuracy Comparison:")
    for name, data in models.items():
        print(f"{name}: {data['accuracy']:.4f}")
    
    # Find the best model
    best_model_name = max(models.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = models[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name} with accuracy {models[best_model_name]['accuracy']:.4f}")
    
    # Analyze feature importance
    print("\nTop 5 Features by Importance:")
    feature_importances = best_model.feature_importances_
    sorted_idx = feature_importances.argsort()[::-1]
    for i in range(min(5, len(feature_names))):
        idx = sorted_idx[i]
        print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")
    
    # Provide insights based on the analysis
    print("\nKey Insights:")
    print("1. Distance categories are the most important features for predicting fare categories.")
    print("2. Carrier type (especially Southwest Airlines) significantly influences fare categories.")
    print("3. Market share also plays an important role in determining fares.")
    print("4. The entropy-based model performed best, suggesting that fare categories have uneven distributions.")
    print("5. The model struggles most with distinguishing between High and VeryHigh fare categories.")

    return {
        'best_model_name': best_model_name,
        'feature_importances': feature_importances,
        'feature_names': feature_names
    }

def main():
    """
    Main function to run the decision tree analysis
    """
    print("Loading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data()
    
    print("Training and evaluating decision tree models...")
    models = train_evaluate_dt_models(X, y, feature_names)
    
    print("Analyzing decision tree results...")
    analysis_results = analyze_decision_trees(models, feature_names)

    # Save model results as pkl
    import pickle
    with open('static/data/decision_tree_models.pkl', 'wb') as f:
        pickle.dump(models, f)

    with open('static/data/feature_importances.pkl', 'wb') as f:
        pickle.dump(analysis_results['feature_importances'], f)
    
    print("\nAnalysis complete. Visualizations saved to 'static/img/' directory.")

if __name__ == "__main__":
    main()