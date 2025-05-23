import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA

if not os.path.exists('static/img/svm'):
    os.makedirs('static/img/svm')
if not os.path.exists('static/data/svm'):
    os.makedirs('static/data/svm')

def load_flight_data(filename='flight_data.csv'):
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_flight_data(df):
    numeric_cols = ['nsmiles', 'passengers', 'fare', 'large_ms']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols + ['carrier_lg'])
    fare_percentiles = df['fare'].quantile([0.25, 0.5, 0.75, 0.9])
    print(f"Fare percentiles: {fare_percentiles.values}")
    threshold = fare_percentiles[0.75]
    df['high_fare'] = (df['fare'] > threshold).astype(int)
    class_counts = df['high_fare'].value_counts()
    print(f"Class distribution: {class_counts}")
    print(f"Class 0: {class_counts[0]}, Class 1: {class_counts.get(1, 0)}")
    if class_counts.get(1, 0) < 0.1 * class_counts[0]:
        print("Warning: Severe class imbalance. Using median as threshold.")
        threshold = df['fare'].median()
        df['high_fare'] = (df['fare'] > threshold).astype(int)
        class_counts = df['high_fare'].value_counts()
        print(f"New class distribution: {class_counts}")
    df['fare_per_mile'] = df['fare'] / df['nsmiles'].replace(0, np.nan)
    df['passenger_density'] = df['passengers'] / df['nsmiles'].replace(0, np.nan)
    return df

def create_svm_visualizations(X, y, output_dir='static/img/svm'):
    print("Creating visualizations of the data before SVM application...")
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
        plt.title('PCA Projection of Flight Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='High Fare (1) or Not (0)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_pca_projection.png'), dpi=300)
        plt.close()
    if X.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
        plt.title('Flight Data - 2D Visualization')
        plt.xlabel(f'Feature 1')
        plt.ylabel(f'Feature 2')
        plt.colorbar(label='High Fare (1) or Not (0)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_2d_visualization.png'), dpi=300)
        plt.close()
    if X.shape[1] >= 2:
        subset_size = min(1000, X.shape[0])
        indices = np.random.choice(X.shape[0], subset_size, replace=False)
        features = ['Distance', 'Passengers', 'Fare', 'Market Share'][:X.shape[1]]
        df_subset = pd.DataFrame(X[indices], columns=features)
        df_subset['High Fare'] = y[indices]
        sns.pairplot(df_subset, hue='High Fare', palette='viridis')
        plt.suptitle('Pair Plot of Flight Features', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_pair_plot.png'), dpi=300)
        plt.close()
    print("Visualizations created successfully.")

def visualize_feature_relationships(df, features, target_col, output_path='static/img/svm/data_pca_projection.png'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    df_viz = df[[*features, target_col]].copy()
    if len(df_viz) > 1000:
        df_viz = (
            df_viz.groupby(target_col, group_keys=False)
            .apply(lambda x: x.sample(int(np.rint(1000 * len(x) / len(df_viz))), random_state=42))
        )
        if len(df_viz) > 1000:
            df_viz = df_viz.sample(n=1000, random_state=42)
    df_viz[target_col] = df_viz[target_col].astype(str)
    sns.pairplot(df_viz, hue=target_col, palette='viridis', diag_kind='kde')
    plt.suptitle('Feature Relationships and Fare Categories', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_decision_boundary(X, y, model, feature_names, title, output_path):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    if hasattr(model, 'decision_function'):
        decision_values = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        decision_values = decision_values.reshape(xx.shape)
        contours = plt.contour(xx, yy, decision_values, levels=[-1, 0, 1], 
                              linestyles=['--', '-', '--'], 
                              colors=['k', 'k', 'k'])
        plt.clabel(contours, inline=1, fontsize=10, fmt={-1: '-1', 0: '0', 1: '+1'})
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    true_counts = np.sum(cm, axis=1)
    pred_counts = np.sum(cm, axis=0)
    plt.xticks([0.5, 1.5], [f'Regular Fare\n(n={pred_counts[0]})', f'High Fare\n(n={pred_counts[1]})'])
    plt.yticks([0.5, 1.5], [f'Regular Fare\n(n={true_counts[0]})', f'High Fare\n(n={true_counts[1]})'])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_kernel_transformations():
    print("Creating visualizations to explain kernel transformations...")
    output_dir = 'static/img/svm'
    np.random.seed(42)
    X = np.random.rand(50, 2) * 4 - 2
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
    plt.title('Original 2D Data (Non-Linearly Separable)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'original_2d_data.png'), dpi=300)
    plt.close()
    X_poly = np.zeros((X.shape[0], 6))
    X_poly[:, 0] = X[:, 0]**2
    X_poly[:, 1] = X[:, 1]**2
    X_poly[:, 2] = np.sqrt(2) * X[:, 0] * X[:, 1]
    X_poly[:, 3] = np.sqrt(2) * X[:, 0]
    X_poly[:, 4] = np.sqrt(2) * X[:, 1]
    X_poly[:, 5] = np.ones(X.shape[0])
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_poly[:, 0], X_poly[:, 1], X_poly[:, 2], c=y, cmap='viridis', s=50, alpha=0.7)
    ax.set_title('Polynomial Kernel Transformation (First 3 Dimensions)')
    ax.set_xlabel('x²')
    ax.set_ylabel('y²')
    ax.set_zlabel('√2·x·y')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'polynomial_transformation.png'), dpi=300)
    plt.close()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_poly[:, 0], X_poly[:, 1], X_poly[:, 2], c=y, cmap='viridis', s=50, alpha=0.7)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                          np.linspace(ylim[0], ylim[1], 10))
    zz = 1 - 0.5*xx - 0.5*yy
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan')
    ax.set_title('Linear Separation in Higher Dimensional Space')
    ax.set_xlabel('x²')
    ax.set_ylabel('y²')
    ax.set_zlabel('√2·x·y')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_separation_in_higher_dim.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
    sv_indices = [np.argmin(np.sum(X[y==0]**2, axis=1)), np.argmin(np.sum(X[y==1]**2, axis=1))]
    sv1 = X[y==0][sv_indices[0]]
    sv2 = X[y==1][sv_indices[1]]
    plt.scatter(sv1[0], sv1[1], s=200, facecolors='none', edgecolors='r', linewidths=2, label='Support Vector 1')
    plt.scatter(sv2[0], sv2[1], s=200, facecolors='none', edgecolors='b', linewidths=2, label='Support Vector 2')
    plt.annotate('Kernel\nTransformation', xy=(0, -1.5), xytext=(0, -2.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 ha='center', va='center', fontsize=12)
    plt.title('SVM Kernel Maps Data to Higher Dimensions for Linear Separation')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kernel_mapping_concept.png'), dpi=300)
    plt.close()
    print("Visualizations created successfully.")

def create_svm_comparison_visualization(results, output_dir='static/img/svm'):
    print("Creating visualizations to compare different SVM models...")
    kernels = list(results.keys())
    c_values = list(results[kernels[0]].keys())
    accuracy_data = []
    for kernel in kernels:
        for c in c_values:
            accuracy_data.append({
                'Kernel': kernel,
                'C Value': c,
                'Accuracy': results[kernel][c]['accuracy'] * 100
            })
    accuracy_df = pd.DataFrame(accuracy_data)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Kernel', y='Accuracy', hue='C Value', data=accuracy_df)
    plt.title('Accuracy Comparison by Kernel and C Value')
    plt.xlabel('Kernel Type')
    plt.ylabel('Accuracy (%)')
    plt.ylim(min(accuracy_df['Accuracy']) - 5, 100)
    for i, p in enumerate(plt.gca().patches):
        plt.gca().annotate(f"{p.get_height():.1f}%",
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 10),
                           textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kernel_c_accuracy_comparison.png'), dpi=300)
    plt.close()
    pivot_table = accuracy_df.pivot(index='Kernel', columns='C Value', values='Accuracy')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.1f')
    plt.title('Accuracy (%) Heatmap by Kernel and C Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=300)
    plt.close()
    print("Comparison visualizations created successfully.")

def run_svm_analysis_with_weights(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train_viz = X_train[:, :2]
    X_test_viz = X_test[:, :2]
    train_sample = pd.DataFrame({
        feature_names[0]: X_train_viz[:5, 0],
        feature_names[1]: X_train_viz[:5, 1],
        'target': y_train[:5]
    })
    train_sample.to_csv('static/data/svm/train_sample_svm.csv', index=False)
    test_sample = pd.DataFrame({
        feature_names[0]: X_test_viz[:5, 0],
        feature_names[1]: X_test_viz[:5, 1],
        'target': y_test[:5]
    })
    test_sample.to_csv('static/data/svm/test_sample_svm.csv', index=False)
    kernels = ['linear', 'poly', 'rbf']
    c_values = [0.1, 1.0, 10.0]
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    output_dir = 'static/img/svm'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = {}
    for kernel in kernels:
        results[kernel] = {}
        for c in c_values:
            svm_model = SVC(kernel=kernel, C=c, random_state=42, cache_size=500, 
                          probability=True, class_weight='balanced')
            svm_model.fit(X_train, y_train)
            y_pred = svm_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            print(f"\n{kernel} kernel, C={c}:")
            print(f"Class 0 F1: {report['0']['f1-score']:.3f}")
            print(f"Class 1 F1: {report['1']['f1-score']:.3f}")
            print(f"Overall accuracy: {accuracy:.3f}")
            cm_title = f'Confusion Matrix - {kernel.capitalize()} Kernel, C={c}'
            cm_output_path = os.path.join(output_dir, f'cm_{kernel}_c{c}.png')
            plot_confusion_matrix(y_test, y_pred, cm_title, cm_output_path)
            results[kernel][c] = {
                'model': svm_model,
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix_path': cm_output_path
            }
            if X.shape[1] >= 2:
                svm_viz = SVC(kernel=kernel, C=c, random_state=42, cache_size=500, 
                            probability=True, class_weight='balanced')
                svm_viz.fit(X_train_viz, y_train)
                db_title = f'Decision Boundary - {kernel.capitalize()} Kernel, C={c}'
                db_output_path = os.path.join(output_dir, f'db_{kernel}_c{c}.png')
                plot_decision_boundary(X_train_viz, y_train, svm_viz, feature_names[:2], db_title, db_output_path)
                results[kernel][c]['decision_boundary_path'] = db_output_path
    create_svm_comparison_visualization(results)
    return results

def find_best_model(results):
    best_accuracy = 0
    best_config = None
    for kernel in results:
        for c in results[kernel]:
            accuracy = results[kernel][c]['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = (kernel, c)
    return best_config, best_accuracy

def perform_svm_analysis():
    df = load_flight_data()
    if df is None:
        print("Error: Could not load flight data.")
        return None
    df_processed = preprocess_flight_data(df)
    if len(df_processed) > 2000:
        print(f"Dataset size: {len(df_processed)}. Sampling 2000 rows for faster computation.")
        df_processed = (
            df_processed.groupby('high_fare', group_keys=False)
            .apply(lambda x: x.sample(int(np.rint(2000 * len(x) / len(df_processed))), random_state=42))
        )
        if len(df_processed) > 2000:
            df_processed = df_processed.sample(n=2000, random_state=42)
    else:
        print(f"Dataset size: {len(df_processed)}. Using all available data.")
    features = ['nsmiles', 'large_ms', 'passengers', 'fare_per_mile']
    X = df_processed[features].values
    y = df_processed['high_fare'].values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_viz = X_scaled[:, :2]
    feature_names = ['Distance (miles)', 'Market Share']
    visualize_kernel_transformations()
    create_svm_visualizations(X_viz, y)
    visualize_feature_relationships(
        df_processed, features, 'high_fare',
        output_path='static/img/svm/data_pca_projection.png'
    )
    results = run_svm_analysis_with_weights(X_scaled, y, feature_names)
    best_config, best_accuracy = find_best_model(results)
    print(f"Best SVM configuration: Kernel={best_config[0]}, C={best_config[1]}")
    print(f"Best accuracy: {best_accuracy * 100:.2f}%")
    with open('static/data/svm/svm_results.pkl', 'wb') as f:
        import pickle
        pickle.dump(results, f)
    with open('static/data/svm/best_model.pkl', 'wb') as f:
        import pickle
        pickle.dump({
            'kernel': best_config[0],
            'C': best_config[1],
            'accuracy': best_accuracy
        }, f)
    return {
        'results': results,
        'best_config': best_config,
        'best_accuracy': best_accuracy
    }

if __name__ == "__main__":
    results = perform_svm_analysis()
    if results:
        print("SVM analysis completed successfully.")
        best_kernel, best_c = results['best_config']
        best_model_results = results['results'][best_kernel][best_c]
        print(f"\nBest Model Configuration:")
        print(f"Kernel: {best_kernel}")
        print(f"C Value: {best_c}")
        print(f"Accuracy: {best_model_results['accuracy'] * 100:.2f}%")
        print("\nClassification Report for Best Model:")
        report = best_model_results['report']
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"Class {class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1-score']:.4f}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        print(f"  Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"  Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
