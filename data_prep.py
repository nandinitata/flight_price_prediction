import pandas as pd
import numpy as np
import plotly.express as px

def create_sample_df():
    """Create a sample dataframe for testing when real data is unavailable"""
    sample_data = {
        'tbl': ['Table1a'] * 5,
        'year': [2024] * 5,
        'quarter': [3] * 5,
        'citymarketid_1': [30257, 30257, 30257, 30257, 30257],
        'citymarketid_2': [32575, 33195, 32575, 32575, 30852],
        'city1': ['Albany, NY'] * 5,
        'city2': ['Los Angeles, CA', 'Tampa, FL', 'Los Angeles, CA', 'Los Angeles, CA', 'Washington, DC'],
        'airportid_1': [10257] * 5,
        'airportid_2': [12954, 15304, 10800, 13891, 10821],
        'airport_1': ['ALB'] * 5,
        'airport_2': ['LGB', 'TPA', 'BUR', 'ONT', 'BWI'],
        'nsmiles': [2468, 1138, 2468, 2468, 325],
        'passengers': [14.24, 149.89, 22.28, 23.26, 135.76],
        'fare': [289.04, 252.35, 302.70, 354.01, 232.20],
        'carrier_lg': ['WN', 'WN', 'WN', 'WN', 'WN'],
        'large_ms': [0.99, 0.66, 0.80, 0.46, 0.99],
        'fare_lg': [288.70, 225.52, 282.82, 304.89, 232.02],
        'carrier_low': ['WN', 'WN', 'WN', 'WN', 'WN'],
        'lf_ms': [0.99, 0.66, 0.80, 0.46, 0.99],
        'fare_low': [288.70, 225.52, 282.82, 304.89, 232.02],
        'tbl1apk': ['202431025712954ALBLGB', '202431025715304ALBTPA', '202431025710800ALBBUR',
                   '202431025713891ALBONT', '202431025710821ALBBWI']
    }
    
    return pd.DataFrame(sample_data)

def load_data(filename='flight_data.csv'):
    """Load and perform initial data cleaning"""
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded {len(df)} rows from {filename}")
    except (FileNotFoundError, IOError) as e:
        print(f"Warning: Could not load file {filename}. Creating sample data. Error: {str(e)}")
        df = create_sample_df()
    
    raw_sample = df.head().to_html(classes='table table-striped table-hover')
    
    cleaned_df = clean_data(df)
    cleaned_sample = cleaned_df.head().to_html(classes='table table-striped table-hover')
    
    return df, cleaned_df, raw_sample, cleaned_sample

def clean_data(df):
    """
    Clean the flight data by:
    1. Dropping rows with missing values
    2. Removing duplicates
    3. Converting numeric columns to appropriate types
    4. Removing outliers using IQR method
    5. Ensuring all text columns are properly formatted strings
    6. Dropping unnecessary columns
    
    Args:
    df (pd.DataFrame): flight data
    
    Returns:
    pd.DataFrame: cleaned flight data
    """
    cleaned_df = df.copy()
    
    required_cols = ['nsmiles', 'passengers', 'fare', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low',
                    'city1', 'city2', 'airport_1', 'airport_2', 'carrier_lg']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")
        for col in missing_cols:
            if col in ['nsmiles', 'passengers', 'fare', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low']:
                cleaned_df[col] = 0.0
            else:
                cleaned_df[col] = 'Unknown'
    
    try:
        cleaned_df['fare_per_mile'] = cleaned_df['fare'] / cleaned_df['nsmiles'].replace(0, np.nan)
        cleaned_df['route_category'] = pd.cut(cleaned_df['nsmiles'], 
                                            bins=[0, 500, 1000, 2000, np.inf],
                                            labels=['Short', 'Medium', 'Long', 'Ultra-Long'])
    except Exception as e:
        print(f"Warning: Could not create derived features: {str(e)}")
        cleaned_df['fare_per_mile'] = 0.0
        cleaned_df['route_category'] = 'Unknown'
    
    essential_cols = ['nsmiles', 'passengers', 'fare', 'carrier_lg']
    essential_cols = [col for col in essential_cols if col in cleaned_df.columns]
    
    if essential_cols:
        cleaned_df = cleaned_df.dropna(subset=essential_cols)
    
    if len(cleaned_df) > 0:
        cleaned_df = cleaned_df.drop_duplicates()
    
    numeric_cols = ['nsmiles', 'passengers', 'fare', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low']
    numeric_cols = [col for col in numeric_cols if col in cleaned_df.columns]
    
    for col in numeric_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    if len(cleaned_df) > 10:
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            cleaned_df = cleaned_df[~((cleaned_df[col] < (Q1 - 1.5 * IQR)) | 
                                    (cleaned_df[col] > (Q3 + 1.5 * IQR)))]
    
    text_cols = ['city1', 'city2', 'airport_1', 'airport_2']
    text_cols = [col for col in text_cols if col in cleaned_df.columns]
    
    if text_cols:
        cleaned_df[text_cols] = cleaned_df[text_cols].astype(str)
    
    cleaned_df = cleaned_df.dropna(subset=essential_cols)
    
    unnecessary_cols = ['tbl', 'tbl1apk']
    for col in unnecessary_cols:
        if col in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(col, axis=1)

    return cleaned_df


def generate_summary_stats(df, cleaned_df):
    """Generate summary statistics"""
    try:
        total_routes = len(df)
        total_carriers = df['carrier_lg'].nunique() if 'carrier_lg' in df.columns else 0
        avg_fare = cleaned_df['fare'].mean() if 'fare' in cleaned_df.columns else 0
        avg_distance = cleaned_df['nsmiles'].mean() if 'nsmiles' in cleaned_df.columns else 0
        total_passengers = cleaned_df['passengers'].sum() if 'passengers' in cleaned_df.columns else 0
        
        if 'carrier_lg' in df.columns and not df['carrier_lg'].empty:
            most_common_carrier = df['carrier_lg'].mode()[0]
        else:
            most_common_carrier = "Unknown"
        
        if 'nsmiles' in df.columns and len(df) > 0:
            longest_route_idx = df['nsmiles'].idxmax() if not df['nsmiles'].isna().all() else 0
            longest_route = {
                'city1': df.loc[longest_route_idx, 'city1'] if 'city1' in df.columns else "Unknown",
                'city2': df.loc[longest_route_idx, 'city2'] if 'city2' in df.columns else "Unknown",
                'nsmiles': df.loc[longest_route_idx, 'nsmiles'] if 'nsmiles' in df.columns else 0
            }
        else:
            longest_route = {'city1': "Unknown", 'city2': "Unknown", 'nsmiles': 0}
        
        if 'fare' in df.columns and len(df) > 0:
            highest_fare_idx = df['fare'].idxmax() if not df['fare'].isna().all() else 0
            highest_fare = {
                'city1': df.loc[highest_fare_idx, 'city1'] if 'city1' in df.columns else "Unknown",
                'city2': df.loc[highest_fare_idx, 'city2'] if 'city2' in df.columns else "Unknown",
                'fare': df.loc[highest_fare_idx, 'fare'] if 'fare' in df.columns else 0
            }
        else:
            highest_fare = {'city1': "Unknown", 'city2': "Unknown", 'fare': 0}
        
    except Exception as e:
        print(f"Error generating summary statistics: {str(e)}")
        total_routes = len(df)
        total_carriers = 0
        avg_fare = 0
        avg_distance = 0
        total_passengers = 0
        most_common_carrier = "Unknown"
        longest_route = {'city1': "Unknown", 'city2': "Unknown", 'nsmiles': 0}
        highest_fare = {'city1': "Unknown", 'city2': "Unknown", 'fare': 0}
    
    summary = {
        'total_routes': total_routes,
        'total_carriers': total_carriers,
        'avg_fare': avg_fare,
        'avg_distance': avg_distance,
        'total_passengers': total_passengers,
        'most_common_carrier': most_common_carrier,
        'longest_route': longest_route,
        'highest_fare': highest_fare
    }
    
    return summary

def create_visualizations(df, cleaned_df):
    """Create all required visualizations with robust error handling"""
    plots = {}
    
    if cleaned_df.empty or len(cleaned_df) < 2:
        for plot_name in ['fare_dist', 'distance_fare', 'market_share', 'route_category', 
                         'fare_box', 'competition', 'fare_efficiency', 'share_fare', 
                         'passenger_vol', 'pax_dist', 'seasonal']:
            fig = px.scatter(x=[0], y=[0])
            fig.update_layout(
                title=f"Insufficient data for {plot_name}",
                annotations=[dict(
                    x=0, y=0,
                    xref="x", yref="y",
                    text="Not enough data for visualization",
                    showarrow=False,
                    font=dict(size=14)
                )]
            )
            plots[plot_name] = fig
        
        return plots
    
    try:
        fig1 = px.histogram(cleaned_df,
                            x='fare',
                            color='carrier_lg',
                            title='Fare Distribution by Carrier')
        plots['fare_dist'] = fig1
    except Exception as e:
        print(f"Error creating fare distribution: {str(e)}")
        plots['fare_dist'] = px.scatter(x=[0], y=[0], title="Error: Fare Distribution")
    
    try:
        fig2 = px.scatter(cleaned_df,
                        x='nsmiles',
                        y='fare',
                        color='carrier_lg',
                        size='passengers',
                        hover_data=['city1', 'city2'],
                        labels={'nsmiles': 'Distance (miles)',
                                'fare': 'Fare ($)',
                                'carrier_lg': 'Carrier'},
                        title='Distance vs Fare by Carrier')
        plots['distance_fare'] = fig2
    except Exception as e:
        print(f"Error creating distance fare plot: {str(e)}")
        plots['distance_fare'] = px.scatter(x=[0], y=[0], title="Error: Distance vs Fare")
    
    try:
        market_share = cleaned_df.groupby('carrier_lg')['large_ms'].mean().reset_index()
        fig3 = px.pie(market_share,
                    values='large_ms',
                    names='carrier_lg',
                    title='Carrier Market Share Distribution')
        plots['market_share'] = fig3
    except Exception as e:
        print(f"Error creating market share plot: {str(e)}")
        plots['market_share'] = px.pie(names=['Error'], values=[1], title="Error: Market Share")
    
    try:
        if 'route_category' in cleaned_df.columns:
            route_cat = cleaned_df.groupby('route_category').agg({
                'fare': 'mean',
                'passengers': 'sum',
                'nsmiles': 'mean'
            }).reset_index()
            
            fig4 = px.bar(route_cat,
                        x='route_category',
                        y=['fare', 'passengers', 'nsmiles'],
                        title='Route Category Analysis',
                        barmode='group')
            plots['route_category'] = fig4
        else:
            plots['route_category'] = px.bar(x=['No Data'], y=[0], title="Route Category Analysis - No Data")
    except Exception as e:
        print(f"Error creating route category plot: {str(e)}")
        plots['route_category'] = px.bar(x=['Error'], y=[0], title="Error: Route Category")
    
    try:
        fig5 = px.box(cleaned_df,
                    x='carrier_lg',
                    y='fare',
                    title='Fare Distribution by Carrier')
        plots['fare_box'] = fig5
    except Exception as e:
        print(f"Error creating fare box plot: {str(e)}")
        plots['fare_box'] = px.box(x=['Error'], y=[0], title="Error: Fare Box Plot")

    try:
        competition = cleaned_df.groupby('city1')['carrier_lg'].nunique().reset_index()
        fig6 = px.bar(competition.sort_values('carrier_lg', ascending=False).head(10),
                    x='city1',
                    y='carrier_lg',
                    title='Number of Carriers by Origin City')
        plots['competition'] = fig6
    except Exception as e:
        print(f"Error creating competition plot: {str(e)}")
        plots['competition'] = px.bar(x=['Error'], y=[0], title="Error: Competition Analysis")

    try:
        if 'fare_per_mile' in cleaned_df.columns and 'route_category' in cleaned_df.columns:
            fig7 = px.scatter(cleaned_df,
                            x='nsmiles',
                            y='fare_per_mile',
                            color='route_category',
                            title='Fare Efficiency (Cost per Mile)')
            plots['fare_efficiency'] = fig7
        else:
            plots['fare_efficiency'] = px.scatter(x=[0], y=[0], title="Fare Efficiency - No Data")
    except Exception as e:
        print(f"Error creating fare efficiency plot: {str(e)}")
        plots['fare_efficiency'] = px.scatter(x=[0], y=[0], title="Error: Fare Efficiency")

    try:
        fig8 = px.scatter(cleaned_df,
                        x='large_ms',
                        y='fare',
                        color='carrier_lg',
                        size='passengers',
                        title='Market Share vs Fare Relationship')
        plots['share_fare'] = fig8
    except Exception as e:
        print(f"Error creating share fare plot: {str(e)}")
        plots['share_fare'] = px.scatter(x=[0], y=[0], title="Error: Market Share vs Fare")

    try:
        passenger_vol = cleaned_df.groupby(['city1', 'city2'])['passengers'].sum().reset_index()
        fig9 = px.bar(passenger_vol.sort_values('passengers', ascending=False).head(10),
                    x='passengers',
                    y='city1',
                    title='Top 10 Routes by Passenger Volume')
        plots['passenger_vol'] = fig9
    except Exception as e:
        print(f"Error creating passenger volume plot: {str(e)}")
        plots['passenger_vol'] = px.bar(x=[0], y=['Error'], title="Error: Passenger Volume")

    try:
        fig10 = px.histogram(cleaned_df,
                            x='passengers',
                            color='carrier_lg',
                            title='Passenger Volume Distribution')
        plots['pax_dist'] = fig10
    except Exception as e:
        print(f"Error creating passenger distribution plot: {str(e)}")
        plots['pax_dist'] = px.histogram(x=[0], title="Error: Passenger Distribution")

    try:
        route_freq = cleaned_df.groupby('city1').size().reset_index(name='frequency')
        fig11 = px.bar(route_freq.sort_values('frequency', ascending=False).head(10),
                    x='city1',
                    y='frequency',
                    title='Top 10 Cities by Flight Frequency')
        plots['route_freq'] = fig11
    except Exception as e:
        print(f"Error creating route frequency plot: {str(e)}")
        plots['route_freq'] = px.bar(x=['Error'], y=[0], title="Error: Route Frequency")

    try:
        if 'quarter' in cleaned_df.columns:
            fig12 = px.box(cleaned_df,
                        x='quarter',
                        y='fare',
                        color='carrier_lg',
                        title='Fare Distribution by Quarter')
            plots['seasonal'] = fig12
        else:
            plots['seasonal'] = px.box(x=['No Data'], y=[0], title="Seasonal Analysis - No Data")
    except Exception as e:
        print(f"Error creating seasonal analysis plot: {str(e)}")
        plots['seasonal'] = px.box(x=['Error'], y=[0], title="Error: Seasonal Analysis")

    return plots

def main():
    raw_df, cleaned_df, _ = load_data()
    
    plots = create_visualizations(raw_df, cleaned_df)
    
    for name, fig in plots.items():
        fig.write_html(f'static/plots/{name}.html')

if __name__ == "__main__":
    main()