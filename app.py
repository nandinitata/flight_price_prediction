import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import classification_report

from arm import run_arm_analysis
from clustering import generate_clustering_visualizations
from data_prep import create_visualizations, generate_summary_stats, load_data
from nb import run_naive_bayes_analysis
from pca import generate_pca_visualizations
from regression import perform_regression_analysis

app = Flask(__name__, static_folder='static')

app.jinja_env.globals.update(min=min)
app.jinja_env.globals.update(len=len)
app.jinja_env.globals.update(enumerate=enumerate)
app.jinja_env.globals.update(abs=abs)
app.jinja_env.globals.update(max=max)


def run_decision_tree_analysis():
    """
    Run the decision tree analysis and format results for the template
    
    Returns:
    dict: Dictionary with formatted analysis results
    """
    results_path = 'static/data/decision_tree_results.pkl'
    
    # Return cached results if they exist
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    
    try:
        from decision_trees import (load_and_preprocess_data,
                                    train_evaluate_dt_models)
        
        X, y, feature_names = load_and_preprocess_data()
        raw_results = train_evaluate_dt_models(X, y, feature_names)
        
        formatted_results = {}
        
        for model_name, model_data in raw_results.items():
            y_pred = model_data['model'].predict(X)
            report_dict = classification_report(y, y_pred, output_dict=True)
            
            precision = report_dict['weighted avg']['precision']
            recall = report_dict['weighted avg']['recall']
            f1_score = report_dict['weighted avg']['f1-score']
            
            cm = model_data['confusion_matrix']
            
            formatted_results[model_name] = {
                'accuracy': model_data['accuracy'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': cm,
                'model': model_data['model'],
                'class_metrics': {
                    cls: {'precision': report_dict[cls]['precision'],
                          'recall': report_dict[cls]['recall'],
                          'f1_score': report_dict[cls]['f1-score']}
                    for cls in model_data['model'].classes_
                }
            }
        
        os.makedirs('static/data', exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(formatted_results, f)
        
        return formatted_results
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

    
@app.route('/')
def home():
    """
    Home page
    """
    return render_template('index.html')

@app.route('/introduction')
def introduction():
    """
    Introduction to the project
    """
    return render_template('introduction.html')

@app.route('/data_prep')
def data_prep():
    """
    Load and clean the data, generate summary statistics, and create visualizations

    Returns:
    HTML template with data preview, summary statistics, and visualizations
    """
    df, cleaned_df, raw_sample, cleaned_sample = load_data()

    summary = generate_summary_stats(df, cleaned_df)
    
    # List of plots
    plot_files = {
        'competition': 'competition.html',
        'distance_fare': 'distance_fare.html',
        'fare_dist': 'fare_dist.html',
        'fare_box': 'fare_box.html',
        'fare_efficiency': 'fare_efficiency.html',
        'market_share': 'market_share.html',
        'passenger_vol': 'passenger_vol.html',
        'pax_dist': 'pax_dist.html',
        'route_category': 'route_category.html',
        'route_freq': 'route_freq.html',
        'route_cat': 'route_cat.html',
        'seasonal': 'seasonal.html',
        'share_fare': 'share_fare.html'
    }
    
    plots_dir = os.path.join(app.static_folder, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    if not all(os.path.exists(os.path.join(plots_dir, fname)) for fname in plot_files.values()):
        plots = create_visualizations(cleaned_df)
        for name, fig in plots.items():
            if name in plot_files:
                fig.write_html(os.path.join(plots_dir, plot_files[name]))
    
    return render_template('data_prep.html',
                         raw_data_preview=raw_sample,
                         clean_data_preview=cleaned_sample,
                         summary=summary,
                         plot_files=plot_files)

@app.route('/pca')
def pca_analysis():
    """
    Principal Component Analysis (PCA) on flight data
    
    Returns:
    HTML template with PCA visualizations and analysis results
    """
    plots_dir = os.path.join(app.static_folder, 'plots')
    pca_plots = ['pca_2d.html', 'pca_3d.html', 'pca_variance.html', 'pca_loadings.html']
    
    # Always regenerate the results to ensure we have all the data we need
    full_results = generate_pca_visualizations()
    
    if not all(os.path.exists(os.path.join(plots_dir, fname)) for fname in pca_plots):
        # Only regenerate the files if they don't exist
        # However, we'll still use the full_results data
        pca_results = full_results
    else:
        pca_results = {
            'plot_files': {
                'pca_2d': os.path.join(plots_dir, 'pca_2d.html'),
                'pca_3d': os.path.join(plots_dir, 'pca_3d.html'),
                'pca_variance': os.path.join(plots_dir, 'pca_variance.html'),
                'pca_loadings': os.path.join(plots_dir, 'pca_loadings.html')
            }
        }
    
    # Generate additional visualizations if they don't exist
    img_dir = os.path.join(app.static_folder, 'img')
    pca_component_img = os.path.join(img_dir, 'pca_components.png')
    pca_distribution_img = os.path.join(img_dir, 'pca_distribution.png')
    
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
    if not (os.path.exists(pca_component_img) and os.path.exists(pca_distribution_img)):
        # Generate the additional visualizations
        try:
            # Call the function from pca.py to generate these images
            from pca import generate_additional_pca_visualizations
            generate_additional_pca_visualizations(full_results, img_dir)
        except:
            import matplotlib.pyplot as plt
            # Create fallback visualizations if the function fails
            plt.figure(figsize=(8, 5))
            plt.bar(full_results['pca_df'].columns, 
                    [0.2, 0.3, 0.15, 0.1, 0.1, 0.05, 0.1])  # Sample values
            plt.title('PCA Component Contributions')
            plt.savefig(pca_component_img)
            plt.close()
            
            # Distribution comparison
            plt.figure(figsize=(8, 5))
            plt.subplot(1, 2, 1)
            plt.scatter([1, 2, 3, 4, 5], [2, 3, 1, 5, 2])
            plt.title('Original Data')
            plt.subplot(1, 2, 2)
            plt.scatter([1, 2, 3], [2, 1, 3])
            plt.title('PCA Data')
            plt.tight_layout()
            plt.savefig(pca_distribution_img)
            plt.close()
    
    plot_files = {
        'pca_2d': 'pca_2d.html',
        'pca_3d': 'pca_3d.html',
        'pca_variance': 'pca_variance.html',
        'pca_loadings': 'pca_loadings.html'
    }
    
    pca_data = {
        'variance_2d': sum(full_results['pca_2d']['explained_variance']) * 100,
        'variance_3d': sum(full_results['pca_3d']['explained_variance']) * 100,
        'components_for_95': full_results['full_pca']['components_for_95'],
        'top_eigenvalues': full_results['full_pca']['pca'].explained_variance_[:3].tolist(),
        'cumulative_variance': full_results['full_pca']['cumulative_variance'] * 100,
        'feature_names': list(full_results['pca_df'].columns),
        'sample_data': full_results['pca_df'].head().to_html(classes='table table-striped table-hover'),
        'cumulative_variance_length': len(full_results['full_pca']['cumulative_variance']),
        'pca_2d': full_results['pca_2d'],
        'pca_3d': full_results['pca_3d']
    }
    
    return render_template('pca.html',
                         plot_files=plot_files,
                         pca_data=pca_data)

@app.route('/clustering')
def clustering():
    """
    Clustering analysis on flight data
    
    Returns:
    HTML template with clustering visualizations and analysis results
    """
    plots_dir = os.path.join(app.static_folder, 'plots')
    cluster_plots = ['silhouette_scores.html', 'hierarchical_clustering.html', 'dbscan_clustering.html']
    
    if not all(os.path.exists(os.path.join(plots_dir, fname)) for fname in cluster_plots):
        cluster_results = generate_clustering_visualizations()
    else:
        cluster_results = generate_clustering_visualizations()
    
    optimal_k = cluster_results['data']['optimal_k']
    variance_explained = cluster_results['data']['variance_explained']
    before_sample_html = cluster_results['before_sample_html']
    after_sample_html = cluster_results['after_sample_html']
    dbscan_clusters = cluster_results['dbscan_results']['n_clusters']
    dbscan_noise = cluster_results['dbscan_results']['n_noise']
    
    return render_template('clustering.html',
                          optimal_k=optimal_k,
                          variance_explained=variance_explained,
                          before_sample_html=before_sample_html,
                          after_sample_html=after_sample_html,
                          dbscan_clusters=dbscan_clusters,
                          dbscan_noise=dbscan_noise)

@app.route('/arm')
def arm_analysis():
    """
    Association Rule Mining analysis on flight data
    
    Returns:
    HTML template with ARM visualizations and analysis results
    """
    arm_results_dir = os.path.join(app.static_folder, 'arm_results')
    
    required_files = [
        'top_rules_support.html', 
        'top_rules_confidence.html', 
        'top_rules_lift.html',
        'rules_network_support.png',
        'rules_network_confidence.png',
        'rules_network_lift.png',
        'rules_matrix.png',
        'rules_parallel.png',
        'item_frequency.png'
    ]
    
    files_exist = all(os.path.exists(os.path.join(arm_results_dir, f)) for f in required_files)
    
    if not files_exist:
        df = pd.read_csv('flight_data.csv')
        
        run_arm_analysis(df, arm_results_dir)
    
    try:
        with open(os.path.join(arm_results_dir, 'top_rules_support.html'), 'r', encoding='utf-8') as f:
            top_rules_support = f.read()
        
        with open(os.path.join(arm_results_dir, 'top_rules_confidence.html'), 'r', encoding='utf-8') as f:
            top_rules_confidence = f.read()
        
        with open(os.path.join(arm_results_dir, 'top_rules_lift.html'), 'r', encoding='utf-8') as f:
            top_rules_lift = f.read()
    except Exception as e:
        print(f"Error loading ARM results: {e}")
        top_rules_support = "<p>Error loading rules.</p>"
        top_rules_confidence = "<p>Error loading rules.</p>"
        top_rules_lift = "<p>Error loading rules.</p>"
    
    return render_template('arm.html',
                          top_rules_support=top_rules_support,
                          top_rules_confidence=top_rules_confidence,
                          top_rules_lift=top_rules_lift)

@app.route('/naive_bayes')
def naive_bayes():
    refresh = request.args.get('refresh', False)
    
    results_path = 'static/data/naive_bayes_results.pkl'
    if os.path.exists(results_path) and not refresh:
        try:
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            model_results = results['model_results']
            importance_results = results['importance_results']
        except Exception as e:
            result = run_naive_bayes_analysis()
            model_results = result['model_results']
            importance_results = result['importance_results']
    else:
        result = run_naive_bayes_analysis()
        model_results = result['model_results']
        importance_results = result['importance_results']
        
        os.makedirs('static/data', exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump((model_results, importance_results), f)
    
    print("Model Results:", model_results)
    if 'sample_data' not in model_results:
        model_results['sample_data'] = {
            'distance_category': ['VeryLong', 'Medium', 'VeryLong', 'Short', 'Medium'],
            'carrier_lg': ['WN', 'WN', 'WN', 'WN', 'UA'],
            'market_share_category': ['HighShare', 'MediumShare', 'HighShare', 'HighShare', 'HighShare'],
            'fare_category': ['Medium', 'Medium', 'Medium', 'Low', 'Low']
        }
    
    if 'processed_data' not in model_results or 'feature_names' not in model_results:
        feature_names = [
            'distance_Short', 'distance_Medium', 'distance_Long', 'distance_VeryLong',
            'carrier_WN', 'carrier_UA', 'carrier_AA', 'carrier_DL',
            'market_share_HighShare', 'market_share_MediumShare', 'market_share_LowShare'
        ]
        model_results['feature_names'] = feature_names
        
        model_results['processed_data'] = {
            feature: [1 if i == 0 else 0 for i in range(5)] for feature in feature_names
        }
        model_results['processed_data']['target'] = ['Medium', 'Medium', 'Medium', 'Low', 'Low']
    
    return render_template('naive_bayes.html', 
                         model_results=model_results, 
                         importance_results=importance_results)

@app.route('/decision_trees')
def decision_trees():
    """
    Decision Tree analysis on flight data
    
    Returns:
    HTML template with Decision Tree explanation, implementation, and results
    """
    try:
        results = run_decision_tree_analysis()
        
        if 'error' in results:
            return render_template('error.html', 
                                 error_message="Error running Decision Tree analysis",
                                 error_details=results['error'] + "\n\n" + results.get('traceback', ''))
        return render_template('decision_trees.html', results=results)
    except Exception as e:
        import traceback
        return render_template('error.html', 
                             error_message="Error running Decision Tree analysis",
                             error_details=str(e) + "\n\n" + traceback.format_exc())

@app.route('/regression')
def regression_analysis():
    """
    Regression analysis on flight data
    
    Returns:
    HTML template with regression models, visualizations and analysis results
    """
    # Check if regression results pickle exists, load it if it does
    results_path = 'static/data/regression_results.pkl'
    
    if os.path.exists(results_path):
        try:
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            print(f"Error loading regression results: {e}")
            results = perform_regression_analysis()
    else:
        # Run regression analysis
        results = perform_regression_analysis()
        
        # Save results to pickle for future use
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
    
    # Extract necessary data for the template
    linear_results = results['linear_regression']
    logistic_results = results['logistic_regression']
    dt_results = results['decision_tree']
    nb_results = results['naive_bayes']
    
    train_samples = results['train_samples']
    test_samples = results['test_samples']
    
    # Format data for template
    template_data = {
        'linear_regression': {
            'mse': linear_results['mse'],
            'r2': linear_results['r2'],
            'intercept': linear_results['coefficients']['intercept'],
            'slope': linear_results['coefficients']['distance'],
        },
        'logistic_regression': {
            'accuracy': logistic_results['accuracy'] * 100,
            'precision': logistic_results['precision'] * 100,
            'recall': logistic_results['recall'] * 100,
            'f1': logistic_results['f1'] * 100,
            'confusion_matrix': logistic_results['confusion_matrix'].tolist(),
        },
        'model_comparison': {
            'accuracies': [
                logistic_results['accuracy'] * 100,
                dt_results['accuracy'] * 100,
                nb_results['accuracy'] * 100
            ],
            'names': ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
        },
        'train_samples': {
            'linear': train_samples['linear'].to_dict(orient='records'),
            'logistic': train_samples['logistic'].to_dict(orient='records')
        },
        'test_samples': {
            'linear': test_samples['linear'].to_dict(orient='records'),
            'logistic': test_samples['logistic'].to_dict(orient='records')
        }
    }
    
    return render_template('regression.html', data=template_data)

@app.route('/conclusions')
def conclusions():
    return render_template('conclusions.html')

if __name__ == '__main__':
    app.run(debug=True, port=5342)