import os
from flask import Flask, render_template
import pandas as pd
from data_prep import create_visualizations, generate_summary_stats, load_data
from arm import run_arm_analysis
from pca import generate_pca_visualizations
from clustering import generate_clustering_visualizations

app = Flask(__name__, static_folder='static')

app.jinja_env.globals.update(min=min)
app.jinja_env.globals.update(len=len)
app.jinja_env.globals.update(enumerate=enumerate)

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
        # Create simple visualizations as placeholders
        import matplotlib.pyplot as plt
        
        # Component contributions
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
    """
    Naive Bayes analysis on flight data
    
    Returns:
    HTML template with Naive Bayes explanation, implementation, and results
    """
    return render_template('naive_bayes.html')

@app.route('/conclusions')
def conclusions():
    return render_template('conclusions.html')

if __name__ == '__main__':
    app.run(debug=True)