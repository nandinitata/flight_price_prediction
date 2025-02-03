import pandas as pd
import numpy as np
import plotly.express as px


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
    
    # Create derived features
    cleaned_df['fare_per_mile'] = cleaned_df['fare'] / cleaned_df['nsmiles']
    cleaned_df['route_category'] = pd.cut(cleaned_df['nsmiles'], 
                                        bins=[0, 500, 1000, 2000, np.inf],
                                        labels=['Short', 'Medium', 'Long', 'Ultra-Long'])
    
    # 1. Drop rows with missing values
    cleaned_df = df.dropna()
    
    # 2. Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 3. Convert numeric columns to appropriate types
    numeric_cols = ['nsmiles', 'passengers', 'fare', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low']
    for col in numeric_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # 4. Remove outliers using IQR method
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        cleaned_df = cleaned_df[~((cleaned_df[col] < (Q1 - 1.5 * IQR)) | 
                                (cleaned_df[col] > (Q3 + 1.5 * IQR)))]
    
    # 5. Ensure all text columns are properly formatted strings
    text_cols = ['city1', 'city2', 'airport_1', 'airport_2']
    cleaned_df[text_cols] = cleaned_df[text_cols].astype(str)
    
    cleaned_df = df.dropna()

    # 6. Drop unnecessary columns
    cleaned_df = cleaned_df.drop(['tbl', 'tbl1apk'], axis=1)

    return cleaned_df

def load_data(filename='src/flight_data.csv'):
    """Load and perform initial data cleaning"""
    df = pd.read_csv(filename)
    
    raw_sample = df.head().to_html(classes='table table-striped table-hover')
    
    # Clean data
    cleaned_df = clean_data(df)
    cleaned_sample = cleaned_df.head().to_html(classes='table table-striped table-hover')
    
    return df, cleaned_df, raw_sample, cleaned_sample

def create_visualizations(df, cleaned_df):
    """Create all required visualizations"""
    plots = {}
    
    # 1. Fare Distribution
    fig1 = px.histogram(cleaned_df,
                          x='fare',
                          color='carrier_lg',
                          title='Fare Distribution by Carrier')
    plots['fare_dist'] = fig1

    # 2. Distance vs Fare Relationship
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

    # 3. Carrier Market Share
    market_share = cleaned_df.groupby('carrier_lg')['large_ms'].mean().reset_index()
    fig3 = px.pie(market_share,
                  values='large_ms',
                  names='carrier_lg',
                  title='Carrier Market Share Distribution')
    plots['market_share'] = fig3

    # 4. Route Category Analysis
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

    # 5. Fare Box Plot
    fig5 = px.box(cleaned_df,
                  x='carrier_lg',
                  y='fare',
                  title='Fare Distribution by Carrier')
    plots['fare_box'] = fig5

    # 6. Market Competition Analysis
    competition = cleaned_df.groupby('city1')['carrier_lg'].nunique().reset_index()
    fig6 = px.bar(competition.sort_values('carrier_lg', ascending=False).head(10),
                  x='city1',
                  y='carrier_lg',
                  title='Number of Carriers by Origin City')
    plots['competition'] = fig6

    # 7. Fare per Mile Analysis
    fig7 = px.scatter(cleaned_df,
                      x='nsmiles',
                      y='fare_per_mile',
                      color='route_category',
                      title='Fare Efficiency (Cost per Mile)')
    plots['fare_efficiency'] = fig7

    # 8. Market Share vs Fare Correlation
    fig8 = px.scatter(cleaned_df,
                      x='large_ms',
                      y='fare',
                      color='carrier_lg',
                      size='passengers',
                      title='Market Share vs Fare Relationship')
    plots['share_fare'] = fig8

    # 9. Top Routes by Passenger Volume
    passenger_vol = cleaned_df.groupby(['city1', 'city2'])['passengers'].sum().reset_index()
    fig9 = px.bar(passenger_vol.sort_values('passengers', ascending=False).head(10),
                  x='passengers',
                  y='city1',
                  title='Top 10 Routes by Passenger Volume')
    plots['passenger_vol'] = fig9

    # 10. Seasonal Analysis
    fig10 = px.box(cleaned_df,
                   x='quarter',
                   y='fare',
                   color='carrier_lg',
                   title='Fare Distribution by Quarter')
    plots['seasonal'] = fig10

    return plots

def generate_summary_stats(df, cleaned_df):
    """Generate summary statistics"""
    summary = {
        'total_routes': len(df),
        'total_carriers': df['carrier_lg'].nunique(),
        'avg_fare': cleaned_df['fare'].mean(),
        'avg_distance': cleaned_df['nsmiles'].mean(),
        'total_passengers': cleaned_df['passengers'].sum(),
        'most_common_carrier': df['carrier_lg'].mode()[0],
        'longest_route': df.loc[df['nsmiles'].idxmax(), ['city1', 'city2', 'nsmiles']].to_dict(),
        'highest_fare': df.loc[df['fare'].idxmax(), ['city1', 'city2', 'fare']].to_dict()
    }
    return summary

def main():
    raw_df, cleaned_df, _ = load_data()
    
    plots = create_visualizations(raw_df, cleaned_df)
    
    # Save figures as HTML files
    for name, fig in plots.items():
        fig.write_html(f'static/plots/{name}.html')

if __name__ == "__main__":
    main()