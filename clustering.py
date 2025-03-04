
import pandas as pd
import numpy as np
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def load_flight_data(filename='flight_data.csv'):
    """Load flight data from CSV file"""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_for_clustering(df):
    """
    Preprocess flight data for clustering analysis:
    1. Select relevant columns
    2. Save labels separately
    3. Keep only numeric features
    4. Normalize the data
    5. Optionally reduce dimensions with PCA
    """
    cluster_df = df.copy()
    
    feature_cols = ['nsmiles', 'passengers', 'fare', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low']
    
    labels = cluster_df['carrier_lg'].copy()
    
    features = cluster_df[feature_cols].copy()
    
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    
    features = features.dropna()
    
    valid_indices = features.index
    labels = labels.loc[valid_indices]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=valid_indices)
    
    before_sample = cluster_df.loc[valid_indices].head(10)
    after_sample = scaled_df.head(10)
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'], index=valid_indices)
    
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    
    return {
        'original_df': cluster_df.loc[valid_indices],
        'features': features,
        'scaled_features': scaled_df,
        'pca_features': pca_df,
        'labels': labels,
        'before_sample': before_sample,
        'after_sample': after_sample,
        'variance_explained': variance_explained,
        'valid_indices': valid_indices
    }

def find_optimal_k(scaled_features, max_k=10):
    """Find optimal k values using silhouette analysis"""
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    best_k = k_values[np.argmax(silhouette_scores)]
    
    filtered_scores = [(k, score) for k, score in zip(k_values, silhouette_scores) 
                      if abs(k - best_k) >= 2]
    second_best = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[0][0]
    
    filtered_scores = [(k, score) for k, score in zip(k_values, silhouette_scores) 
                      if k != best_k and k != second_best]
    third_best = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[0][0]
    
    optimal_k = [best_k, second_best, third_best]
    optimal_k.sort()  
    
    fig = px.line(
        x=list(k_values),
        y=silhouette_scores,
        markers=True,
        title='Silhouette Score for Different Values of k',
        labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'}
    )
    
    for k in optimal_k:
        fig.add_vline(x=k, line_dash="dash", line_color="red")
        
    fig.add_annotation(
        x=optimal_k[0],
        y=silhouette_scores[optimal_k[0]-2],
        text=f"k={optimal_k[0]}",
        showarrow=True,
        arrowhead=1
    )
    
    return {
        'optimal_k': optimal_k,
        'silhouette_plot': fig,
        'silhouette_scores': silhouette_scores
    }

def kmeans_clustering(data, optimal_k):
    """
    Perform K-means clustering with optimal k values and create visualizations
    """
    scaled_features = data['scaled_features'].values
    pca_features = data['pca_features'].values
    labels = data['labels']
    
    kmeans_results = {}
    
    for k in optimal_k:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        centroids = kmeans.cluster_centers_
        pca = PCA(n_components=3)
        pca.fit(scaled_features)
        centroids_pca = pca.transform(centroids)
        
        fig1 = px.scatter_3d(
            data['pca_features'],
            x='PC1',
            y='PC2',
            z='PC3',
            color=labels,
            title=f'K-means Clustering (k={k}) Colored by Original Carrier',
            labels={'color': 'Carrier'}
        )
        
        fig1.add_trace(
            go.Scatter3d(
                x=centroids_pca[:, 0],
                y=centroids_pca[:, 1],
                z=centroids_pca[:, 2],
                mode='markers',
                marker=dict(
                    color='black',
                    size=10,
                    symbol='diamond'
                ),
                name='Centroids'
            )
        )
        
        fig2 = px.scatter_3d(
            data['pca_features'],
            x='PC1',
            y='PC2',
            z='PC3',
            color=cluster_labels.astype(str),
            title=f'K-means Clustering (k={k}) Colored by Cluster',
            labels={'color': 'Cluster'}
        )
        
        fig2.add_trace(
            go.Scatter3d(
                x=centroids_pca[:, 0],
                y=centroids_pca[:, 1],
                z=centroids_pca[:, 2],
                mode='markers',
                marker=dict(
                    color='black',
                    size=10,
                    symbol='diamond'
                ),
                name='Centroids'
            )
        )
        
        kmeans_results[k] = {
            'model': kmeans,
            'cluster_labels': cluster_labels,
            'centroids': centroids,
            'centroids_pca': centroids_pca,
            'plot_by_carrier': fig1,
            'plot_by_cluster': fig2
        }
    
    return kmeans_results

def hierarchical_clustering(data):
    """
    Perform hierarchical clustering and create dendrogram visualization
    """
    pca_features = data['pca_features'].values
    n_samples = min(1000, len(pca_features)) 
    
    if len(pca_features) > n_samples:
        np.random.seed(42)
        sampled_indices = np.random.choice(len(pca_features), n_samples, replace=False)
        sampled_features = pca_features[sampled_indices]
        sampled_labels = data['labels'].iloc[sampled_indices]
    else:
        sampled_features = pca_features
        sampled_labels = data['labels']
    
    linkage_matrix = linkage(sampled_features, method='ward')
    
    fig = ff.create_dendrogram(
        sampled_features,
        orientation='left',
        labels=sampled_labels.values,
        linkagefun=lambda x: linkage_matrix
    )
    
    fig.update_layout(
        title='Hierarchical Clustering Dendrogram',
        width=800,
        height=600
    )
    
    optimal_k = data.get('optimal_k', [3, 5, 7])
    k = optimal_k[1]
    hierarchical_labels = fcluster(linkage_matrix, k, criterion='maxclust')
    
    fig2 = px.scatter_3d(
        pd.DataFrame(sampled_features, columns=['PC1', 'PC2', 'PC3']),
        x='PC1',
        y='PC2',
        z='PC3',
        color=hierarchical_labels.astype(str),
        title=f'Hierarchical Clustering (k={k})',
        labels={'color': 'Cluster'}
    )
    
    return {
        'linkage_matrix': linkage_matrix,
        'dendrogram': fig,
        'hierarchical_labels': hierarchical_labels,
        'plot': fig2
    }

def dbscan_clustering(data):
    """
    Perform DBSCAN clustering and create visualization
    """
    pca_features = data['pca_features'].values
    
    eps = 0.5 
    min_samples = 5  
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(pca_features)
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    fig = px.scatter_3d(
        data['pca_features'],
        x='PC1',
        y='PC2',
        z='PC3',
        color=dbscan_labels.astype(str),
        title=f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})',
        labels={'color': 'Cluster'}
    )
    
    return {
        'model': dbscan,
        'cluster_labels': dbscan_labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'plot': fig
    }

def generate_clustering_visualizations(save_dir='static/plots'):
    """
    Generate all clustering visualizations and save them to files
    
    Returns:
        dict: Dictionary with clustering results and file paths
    """

    if not os.path.exists('static/data'):
        os.makedirs('static/data')

    if os.path.exists('static/data/clustering_results.pkl'):
        with open('static/data/clustering_results.pkl', 'rb') as f:
            return pickle.load(f)
        
    df = load_flight_data()
    if df is None:
        return None
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data = preprocess_for_clustering(df)
    
    silhouette_results = find_optimal_k(data['scaled_features'].values)
    data['optimal_k'] = silhouette_results['optimal_k']
    
    kmeans_results = kmeans_clustering(data, data['optimal_k'])
    
    hierarchical_results = hierarchical_clustering(data)
    
    dbscan_results = dbscan_clustering(data)
    
    plot_files = {}
    
    plot_files['silhouette'] = os.path.join(save_dir, 'silhouette_scores.html')
    silhouette_results['silhouette_plot'].write_html(plot_files['silhouette'])
    
    for k, result in kmeans_results.items():
        plot_files[f'kmeans_{k}_carrier'] = os.path.join(save_dir, f'kmeans_{k}_carrier.html')
        result['plot_by_carrier'].write_html(plot_files[f'kmeans_{k}_carrier'])
        
        plot_files[f'kmeans_{k}_cluster'] = os.path.join(save_dir, f'kmeans_{k}_cluster.html')
        result['plot_by_cluster'].write_html(plot_files[f'kmeans_{k}_cluster'])
    
    plot_files['hierarchical'] = os.path.join(save_dir, 'hierarchical_clustering.html')
    hierarchical_results['dendrogram'].write_html(plot_files['hierarchical'])
    
    plot_files['hierarchical_3d'] = os.path.join(save_dir, 'hierarchical_clustering_3d.html')
    hierarchical_results['plot'].write_html(plot_files['hierarchical_3d'])
    
    plot_files['dbscan'] = os.path.join(save_dir, 'dbscan_clustering.html')
    dbscan_results['plot'].write_html(plot_files['dbscan'])
    
    before_sample_html = data['before_sample'].to_html(classes='table table-striped table-hover')
    after_sample_html = data['after_sample'].to_html(classes='table table-striped table-hover')
    
    cluster_results = {
        'data': data,
        'silhouette_results': silhouette_results,
        'kmeans_results': kmeans_results,
        'hierarchical_results': hierarchical_results,
        'dbscan_results': dbscan_results,
        'plot_files': plot_files,
        'before_sample_html': before_sample_html,
        'after_sample_html': after_sample_html
    }

    with open('static/data/clustering_results.pkl', 'wb') as f:
        pickle.dump(cluster_results, f)

    return cluster_results


if __name__ == "__main__":
    results = generate_clustering_visualizations()
    
    if results:
        print("Clustering visualizations generated successfully.")
        print(f"Number of data points: {len(results['data']['original_df'])}")
        print(f"Optimal k values: {results['data']['optimal_k']}")
        print(f"PCA variance explained: {results['data']['variance_explained']:.2f}%")
        print(f"DBSCAN clusters found: {results['dbscan_results']['n_clusters']}")
        print(f"DBSCAN noise points: {results['dbscan_results']['n_noise']}")