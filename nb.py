import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path='flight_data.csv'):
    """
    Load flight data and preprocess it for Naive Bayes classification
    """
    df = pd.read_csv(file_path)
    
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
    
    sample_original = df[['distance_category', 'carrier_lg', 'market_share_category', 'fare_category']].head(10)
    sample_original.to_csv('sample_original.csv', index=False)
    
    features = df[['distance_category', 'carrier_lg', 'market_share_category']]
    
    target = df['fare_category']
    
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded_features = encoder.fit_transform(features)
    
    feature_names = []
    for i, col in enumerate(['distance_category', 'carrier_lg', 'market_share_category']):
        for cat in encoder.categories_[i]:
            feature_names.append(f"{col}_{cat}")
    
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
    
    sample_processed = encoded_df.head(10)
    sample_processed['target'] = target.head(10)
    sample_processed.to_csv('sample_processed.csv', index=False)
    
    return encoded_features, target, feature_names, encoder.categories_

def train_evaluate_nb_models(X, y, feature_names):
    """
    Train and evaluate both Multinomial and Bernoulli Naive Bayes models
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    mnb = MultinomialNB(alpha=1.0)  # alpha=1.0 is Laplace smoothing
    mnb.fit(X_train, y_train)
    
    mnb_predictions = mnb.predict(X_test)
    
    mnb_accuracy = accuracy_score(y_test, mnb_predictions)
    mnb_conf_matrix = confusion_matrix(y_test, mnb_predictions)
    mnb_report = classification_report(y_test, mnb_predictions, output_dict=True)
    
    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(X_train, y_train)
    
    bnb_predictions = bnb.predict(X_test)
    
    bnb_accuracy = accuracy_score(y_test, bnb_predictions)
    bnb_conf_matrix = confusion_matrix(y_test, bnb_predictions)
    bnb_report = classification_report(y_test, bnb_predictions, output_dict=True)
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
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
    
    return {
        'multinomial': {
            'accuracy': mnb_accuracy,
            'confusion_matrix': mnb_conf_matrix,
            'report': mnb_report,
            'model': mnb
        },
        'bernoulli': {
            'accuracy': bnb_accuracy,
            'confusion_matrix': bnb_conf_matrix,
            'report': bnb_report,
            'model': bnb
        }
    }

def analyze_feature_importance(models, feature_names):
    """
    Analyze which features are most important for each class in the Naive Bayes models
    """
    results = {}
    
    for model_name, model_data in models.items():
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

if __name__ == "__main__":
    X, y, feature_names, categories = load_and_preprocess_data()
    
    model_results = train_evaluate_nb_models(X, y, feature_names)
    
    importance_results = analyze_feature_importance(model_results, feature_names)
    
    for model_name, model_data in model_results.items():
        print(f"\n{model_name.capitalize()} Naive Bayes Results:")
        print(f"Accuracy: {model_data['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(model_data['confusion_matrix'])
        print("\nClassification Report:")
        print(classification_report(y, model_data['model'].predict(X)))
        
        print("\nMost Important Features by Class:")
        for cls, importance in importance_results[model_name].items():
            print(f"\nClass: {cls}")
            for feature, value in zip(importance['features'], importance['values']):
                print(f"  {feature}: {value:.4f}")