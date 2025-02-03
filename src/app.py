import os
from flask import Flask, render_template
import pandas as pd

from data_prep import create_visualizations, generate_summary_stats, load_data

app = Flask(__name__, static_folder='static')

# Routes
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


@app.route('/conclusions')
def conclusions():
    return render_template('conclusions.html')

if __name__ == '__main__':
    app.run(debug=True)