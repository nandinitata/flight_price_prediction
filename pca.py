import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def load_flight_data(filename='flight_data.csv'):
    """Load flight data from CSV file"""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_for_pca(df):
    """
    Preprocess flight data for PCA analysis:
    1. Select only numeric columns
    2. Drop any rows with missing values
    3. Remove non-numeric or identifier columns
    """
    pca_df = df.copy()
    
    numeric_cols = ['nsmiles', 'passengers', 'fare', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low']
    
    for col in numeric_cols:
        pca_df[col] = pd.to_numeric(pca_df[col], errors='coerce')
    
    pca_df = pca_df[numeric_cols].dropna()
    
    return pca_df

def perform_pca_analysis(df, n_components=None):
    """
    Perform PCA analysis on the flight data
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe with numeric columns
        n_components (int, optional): Number of components to keep. If None, keep all.
        
    Returns:
        dict: Dictionary with PCA results
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    if n_components is None:
        n_components = min(df.shape)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    components_for_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return {
        'pca': pca,
        'pca_result': pca_result,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'components_for_95': components_for_95,
        'loadings': loadings,
        'feature_names': df.columns,
        'scaled_data': scaled_data
    }

def create_2d_plot(pca_result, explained_variance):
    """Create 2D scatter plot of PCA results"""
    variance_x = explained_variance[0] * 100
    variance_y = explained_variance[1] * 100
    
    fig = px.scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        title='PCA: 2D Projection',
        labels={
            'x': f'PC1 ({variance_x:.2f}%)',
            'y': f'PC2 ({variance_y:.2f}%)'
        },
        color_discrete_sequence=['#636EFA'],
        opacity=0.7
    )
    
    fig.add_shape(
        type='line',
        x0=0, y0=min(pca_result[:, 1]) - 1,
        x1=0, y1=max(pca_result[:, 1]) + 1,
        line=dict(color='gray', width=1, dash='dash')
    )
    
    fig.add_shape(
        type='line',
        x0=min(pca_result[:, 0]) - 1, y0=0,
        x1=max(pca_result[:, 0]) + 1, y1=0,
        line=dict(color='gray', width=1, dash='dash')
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        width=800,
        height=600,
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_xaxes(
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig

def create_3d_plot(pca_result, explained_variance):
    """Create 3D scatter plot of PCA results"""
    variance_x = explained_variance[0] * 100
    variance_y = explained_variance[1] * 100
    variance_z = explained_variance[2] * 100
    
    fig = px.scatter_3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        title='PCA: 3D Projection',
        labels={
            'x': f'PC1 ({variance_x:.2f}%)',
            'y': f'PC2 ({variance_y:.2f}%)',
            'z': f'PC3 ({variance_z:.2f}%)'
        },
        color_discrete_sequence=['#636EFA'],
        opacity=0.7
    )
    
    fig.update_layout(
        width=800,
        height=700,
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_variance_plot(explained_variance):
    """Create scree plot of explained variance"""
    cumulative_variance = np.cumsum(explained_variance) * 100
    individual_variance = explained_variance * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(individual_variance) + 1)),
            y=individual_variance,
            name='Individual Variance (%)',
            marker_color='rgb(55, 83, 109)'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance,
            name='Cumulative Variance (%)',
            marker_color='rgb(219, 64, 82)',
            line=dict(width=3)
        ),
        secondary_y=True
    )
    
    fig.add_shape(
        type='line',
        x0=0.5, y0=95,
        x1=len(individual_variance) + 0.5, y1=95,
        line=dict(color='green', width=2, dash='dash'),
        yref='y2'
    )
    
    fig.add_annotation(
        x=len(individual_variance) * 0.8,
        y=96,
        text='95% Variance Threshold',
        showarrow=False,
        font=dict(color='green'),
        yref='y2'
    )
    
    fig.update_layout(
        title_text='Explained Variance by Principal Component',
        xaxis_title='Principal Component',
        yaxis_title='Individual Variance (%)',
        yaxis2_title='Cumulative Variance (%)',
        yaxis2=dict(range=[0, 105]),
        plot_bgcolor='white',
        width=800,
        height=500,
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        dtick=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        secondary_y=False
    )
    
    fig.update_yaxes(
        showgrid=False,
        secondary_y=True
    )
    
    return fig

def create_loadings_plot(pca_result, loadings, feature_names):
    """Create biplot of PCA loadings"""
    fig = px.scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        title='PCA Biplot: Feature Loadings',
        opacity=0.5
    )
    
    scale = 1.0 / max(np.max(np.abs(loadings[:, 0])), np.max(np.abs(loadings[:, 1]))) * 5
    
    for i, feature in enumerate(feature_names):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0] * scale,
            y1=loadings[i, 1] * scale,
            line=dict(color='red', width=2),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=loadings[i, 0] * scale * 1.1,
            y=loadings[i, 1] * scale * 1.1,
            text=feature,
            showarrow=False,
            font=dict(color='red', size=10)
        )
    
    fig.update_layout(
        plot_bgcolor='white',
        width=800,
        height=600,
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_xaxes(
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig

def generate_pca_visualizations(save_dir='static/plots'):
    """
    Generate all PCA visualizations and save them to files
    
    Returns:
        dict: Dictionary with PCA results and file paths
    """
    df = load_flight_data()
    if df is None:
        return None
    
    pca_df = preprocess_for_pca(df)
    
    full_pca_results = perform_pca_analysis(pca_df)
    
    pca_2d_results = perform_pca_analysis(pca_df, n_components=2)
    
    pca_3d_results = perform_pca_analysis(pca_df, n_components=3)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plot_files = {}
    
    fig_2d = create_2d_plot(pca_2d_results['pca_result'], pca_2d_results['explained_variance'])
    plot_files['pca_2d'] = os.path.join(save_dir, 'pca_2d.html')
    fig_2d.write_html(plot_files['pca_2d'])
    
    fig_3d = create_3d_plot(pca_3d_results['pca_result'], pca_3d_results['explained_variance'])
    plot_files['pca_3d'] = os.path.join(save_dir, 'pca_3d.html')
    fig_3d.write_html(plot_files['pca_3d'])
    
    fig_variance = create_variance_plot(full_pca_results['explained_variance'])
    plot_files['pca_variance'] = os.path.join(save_dir, 'pca_variance.html')
    fig_variance.write_html(plot_files['pca_variance'])
    
    fig_loadings = create_loadings_plot(
        pca_2d_results['pca_result'],
        pca_2d_results['loadings'][:, :2],
        pca_2d_results['feature_names']
    )
    plot_files['pca_loadings'] = os.path.join(save_dir, 'pca_loadings.html')
    fig_loadings.write_html(plot_files['pca_loadings'])
    
    return {
        'full_pca': full_pca_results,
        'pca_2d': pca_2d_results,
        'pca_3d': pca_3d_results,
        'plot_files': plot_files,
        'pca_df': pca_df
    }

if __name__ == "__main__":
    results = generate_pca_visualizations()
    
    if results:
        print("PCA visualizations generated successfully.")
        print(f"2D variance explained: {np.sum(results['pca_2d']['explained_variance']) * 100:.2f}%")
        print(f"3D variance explained: {np.sum(results['pca_3d']['explained_variance']) * 100:.2f}%")
        print(f"Components needed for 95% variance: {results['full_pca']['components_for_95']}")
        print(f"Top 3 eigenvalues: {results['full_pca']['pca'].explained_variance_[:3]}")