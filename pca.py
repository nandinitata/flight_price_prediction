import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def create_distribution_comparison_viz(original_data, pca_result):
    """Create visualization comparing data distribution before and after PCA"""
    # Create a figure with 1 row and 2 columns for the comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original Data Distribution", "PCA-Transformed Data"),
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}]]
    )
    
    # Sample points for better visualization
    n_samples = min(1000, len(original_data))
    if len(original_data) > n_samples:
        np.random.seed(42)
        indices = np.random.choice(len(original_data), n_samples, replace=False)
        original_sample = original_data.iloc[indices]
        pca_sample = pca_result[indices]
    else:
        original_sample = original_data
        pca_sample = pca_result
    
    # Original data scatter plot (using the first 2 features for visualization)
    feature1 = original_data.columns[0]
    feature2 = original_data.columns[1]
    
    fig.add_trace(
        go.Scatter(
            x=original_sample[feature1],
            y=original_sample[feature2],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
                opacity=0.7
            ),
            name='Original Features'
        ),
        row=1, col=1
    )
    
    # PCA data scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=pca_sample[:, 0],
            y=pca_sample[:, 1],
            z=pca_sample[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=pca_sample[:, 0],  # Color by PC1 value
                colorscale='Viridis',
                opacity=0.7
            ),
            name='PCA Features'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Data Distribution: Original vs PCA-Transformed",
        width=900,
        height=450,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=feature1, row=1, col=1)
    fig.update_yaxes(title_text=feature2, row=1, col=1)
    
    # Update 3D scene
    fig.update_scenes(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3',
        row=1, col=2
    )
    
    return fig

def generate_additional_pca_visualizations(results, save_dir='static/img'):
    """Generate and save additional PCA visualizations"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create component contributions visualization
    fig_components = create_component_contribution_viz(
        results['pca_3d'], 
        results['pca_df'].columns
    )
    components_path = os.path.join(save_dir, 'pca_components.png')
    fig_components.write_image(components_path)
    
    # Create distribution comparison visualization
    fig_distribution = create_distribution_comparison_viz(
        results['pca_df'],
        results['pca_3d']['pca_result']
    )
    distribution_path = os.path.join(save_dir, 'pca_distribution.png')
    fig_distribution.write_image(distribution_path)
    
    # Create a dynamic 3D loadings plot
    fig_3d_loadings = create_3d_loadings_plot(
        results['pca_3d']['loadings'], 
        results['pca_df'].columns
    )
    loadings_3d_path = os.path.join(save_dir, 'pca_3d_loadings.png')
    fig_3d_loadings.write_image(loadings_3d_path)
    
    # Create explained variance per feature chart
    fig_feature_variance = create_feature_variance_plot(
        results['pca_3d']['pca'], 
        results['pca_df'].columns
    )
    feature_variance_path = os.path.join(save_dir, 'pca_feature_variance.png')
    fig_feature_variance.write_image(feature_variance_path)
    
    return {
        'components_plot': components_path,
        'distribution_plot': distribution_path,
        'loadings_3d_plot': loadings_3d_path,
        'feature_variance_plot': feature_variance_path
    }

def create_3d_loadings_plot(loadings, feature_names):
    """Create a 3D visualization of feature loadings on first 3 principal components"""
    # Extract loadings for first 3 components
    loadings_3d = loadings[:, :3]
    
    # Create the 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=loadings_3d[:, 0],
            y=loadings_3d[:, 1],
            z=loadings_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color=np.sum(np.abs(loadings_3d), axis=1),  # Color by total loading magnitude
                colorscale='Viridis',
                opacity=0.8
            ),
            text=feature_names,
            hoverinfo='text',
            textposition="top center",
            textfont=dict(size=10)
        )
    ])
    
    # Add coordinate system lines
    for i, j, k, color in [
        ([0, 0], [0, 0], [-1, 1], 'blue'),  # Z-axis
        ([0, 0], [-1, 1], [0, 0], 'green'),  # Y-axis
        ([-1, 1], [0, 0], [0, 0], 'red')     # X-axis
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=i, y=j, z=k,
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False
            )
        )
    
    # Add labels for axes
    for i, text, pos in [
        (0, 'PC1', [1.1, 0, 0]),
        (1, 'PC2', [0, 1.1, 0]),
        (2, 'PC3', [0, 0, 1.1])
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='text',
                text=[text],
                textposition="middle center",
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title='3D Feature Loadings on Principal Components',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=800,
        height=700
    )
    
    return fig

def create_feature_variance_plot(pca, feature_names):
    """Create a visualization of how much variance each original feature contributes"""
    # Get component loadings
    components = pca.components_
    
    # Calculate the explained variance contributed by each feature
    explained_variance = pca.explained_variance_ratio_
    
    # Calculate feature importance as the sum of absolute loadings weighted by explained variance
    feature_importance = np.zeros(len(feature_names))
    for i in range(len(components)):
        feature_importance += np.abs(components[i]) * explained_variance[i]
    
    # Normalize to percentage
    feature_importance = feature_importance / np.sum(feature_importance) * 100
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=sorted_importance,
        y=sorted_features,
        orientation='h',
        marker=dict(
            color=sorted_importance,
            colorscale='Viridis',
            colorbar=dict(title='Importance (%)')
        ),
        text=[f"{val:.1f}%" for val in sorted_importance],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Feature Importance in PCA Variance',
        xaxis_title='Contribution to Explained Variance (%)',
        yaxis_title='Original Features',
        plot_bgcolor='white',
        width=800,
        height=500,
        yaxis=dict(
            categoryorder='array',
            categoryarray=sorted_features[::-1]  # Reverse to get descending order
        )
    )
    
    return fig

def create_component_contribution_viz(pca_results, feature_names):
    """Create visualization of how original features contribute to principal components"""
    loadings = pca_results['loadings'][:, :3]  # Get loadings for first 3 PCs
    
    # Calculate absolute contribution to each PC
    abs_loadings = np.abs(loadings)
    contributions = abs_loadings / abs_loadings.sum(axis=0) * 100
    
    # Create a grouped bar chart of contributions
    fig = go.Figure()
    
    for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
        fig.add_trace(go.Bar(
            x=feature_names,
            y=contributions[:, i],
            name=pc,
            text=[f"{val:.1f}%" for val in contributions[:, i]],
            textposition="auto",
            marker_color=['#636EFA', '#EF553B', '#00CC96'][i]
        ))
    
    fig.update_layout(
        title='Feature Contributions to Principal Components',
        xaxis_title='Original Features',
        yaxis_title='Contribution (%)',
        barmode='group',
        legend_title="Principal Component",
        plot_bgcolor='white',
        width=800,
        height=500,
    )
    
    return fig

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
    
    # Generate additional visualizations
    additional_viz = generate_additional_pca_visualizations(results)
    
    print("Additional PCA visualizations generated successfully:")
    for name, path in additional_viz.items():
        print(f"- {name}: {path}")
    if results:
        print("PCA visualizations generated successfully.")
        print(f"2D variance explained: {np.sum(results['pca_2d']['explained_variance']) * 100:.2f}%")
        print(f"3D variance explained: {np.sum(results['pca_3d']['explained_variance']) * 100:.2f}%")
        print(f"Components needed for 95% variance: {results['full_pca']['components_for_95']}")
        print(f"Top 3 eigenvalues: {results['full_pca']['pca'].explained_variance_[:3]}")