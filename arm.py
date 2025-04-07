import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def prepare_flight_data_for_arm(df):
    """
    Prepare flight data for association rule mining by creating categorical features
    and transforming into transaction format
    """
    flight_data = df.copy()
    
    flight_data['fare_category'] = pd.cut(
        flight_data['fare'],
        bins=[0, 150, 250, 350, 500, float('inf')],
        labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
    )
    
    flight_data['distance_category'] = pd.cut(
        flight_data['nsmiles'],
        bins=[0, 500, 1000, 2000, float('inf')],
        labels=['Short', 'Medium', 'Long', 'VeryLong']
    )
    
    flight_data['ms_category'] = pd.cut(
        flight_data['large_ms'],
        bins=[0, 0.33, 0.66, 1],
        labels=['LowShare', 'MediumShare', 'HighShare']
    )
    
    for col in ['fare', 'nsmiles', 'large_ms']:
        flight_data[col] = pd.to_numeric(flight_data[col], errors='coerce')
    
    flight_data = flight_data.dropna(subset=['city1', 'city2', 'carrier_lg', 'fare_category', 'distance_category', 'ms_category'])
    
    transactions = []
    
    for idx, row in flight_data.iterrows():
        transaction = [
            f"Origin_{row['city1']}",
            f"Dest_{row['city2']}",
            f"Carrier_{row['carrier_lg']}",
            f"Fare_{row['fare_category']}",
            f"Distance_{row['distance_category']}",
            f"MarketShare_{row['ms_category']}"
        ]
        transactions.append(transaction)
    
    sample_transactions = transactions[:20]
    
    return {
        'transactions': transactions,
        'sample_transactions': sample_transactions,
        'num_transactions': len(transactions),
        'original_data': flight_data
    }

def create_one_hot_encoded(transactions):
    """Convert transactions to one-hot encoded format for mlxtend"""
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

def generate_item_frequency_plot(transactions, output_dir):
    """Generate a bar chart of item frequencies"""
    all_items = [item for sublist in transactions for item in sublist]
    
    item_counts = pd.Series(all_items).value_counts()
    
    top_items = item_counts.head(20)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(top_items.index, top_items.values, color='skyblue')
    
    plt.title('Top 20 Most Frequent Items', fontsize=16)
    plt.xlabel('Items', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'item_frequency.png'))
    plt.close()

def perform_arm_analysis(transactions, output_dir):
    """
    Perform association rule mining analysis with different thresholds
    and save results and visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_encoded = create_one_hot_encoded(transactions)
    
    generate_item_frequency_plot(transactions, output_dir)
    
    frequent_itemsets_high_support = apriori(df_encoded, min_support=0.1, use_colnames=True)
    
    frequent_itemsets_medium_support = apriori(df_encoded, min_support=0.05, use_colnames=True)
    
    frequent_itemsets_low_support = apriori(df_encoded, min_support=0.02, use_colnames=True)
    
    rules_by_support = association_rules(frequent_itemsets_high_support, metric="support", min_threshold=0.1)
    rules_by_confidence = association_rules(frequent_itemsets_medium_support, metric="confidence", min_threshold=0.7)
    rules_by_lift = association_rules(frequent_itemsets_low_support, metric="lift", min_threshold=1.2)
    
    rules_by_support = rules_by_support.sort_values('support', ascending=False)
    rules_by_confidence = rules_by_confidence.sort_values('confidence', ascending=False)
    rules_by_lift = rules_by_lift.sort_values('lift', ascending=False)
    
    top_rules_support = rules_by_support.head(15)
    top_rules_confidence = rules_by_confidence.head(15)
    top_rules_lift = rules_by_lift.head(15)
    
    support_table = create_rules_html_table(top_rules_support, 'support')
    confidence_table = create_rules_html_table(top_rules_confidence, 'confidence')
    lift_table = create_rules_html_table(top_rules_lift, 'lift')
    
    with open(os.path.join(output_dir, 'top_rules_support.html'), 'w', encoding='utf-8') as f:
        f.write(support_table)

    with open(os.path.join(output_dir, 'top_rules_confidence.html'), 'w', encoding='utf-8') as f:
        f.write(confidence_table)

    with open(os.path.join(output_dir, 'top_rules_lift.html'), 'w', encoding='utf-8') as f:
        f.write(lift_table)
    
    create_network_visualization(top_rules_support, os.path.join(output_dir, 'rules_network_support.png'), 'Association Rules Network (Top by Support)')
    create_network_visualization(top_rules_confidence, os.path.join(output_dir, 'rules_network_confidence.png'), 'Association Rules Network (Top by Confidence)')
    create_network_visualization(top_rules_lift, os.path.join(output_dir, 'rules_network_lift.png'), 'Association Rules Network (Top by Lift)')
    
    create_matrix_visualization(top_rules_lift, os.path.join(output_dir, 'rules_matrix.png'))
    create_parallel_coordinates(top_rules_lift, os.path.join(output_dir, 'rules_parallel.png'))
    
    return {
        'num_rules_support': len(rules_by_support),
        'num_rules_confidence': len(rules_by_confidence),
        'num_rules_lift': len(rules_by_lift),
        'top_support_threshold': 0.1,
        'top_confidence_threshold': 0.7,
        'top_lift_threshold': 1.2
    }

def create_rules_html_table(rules_df, metric_name):
    """Create an HTML table from a rules DataFrame"""
    rules_df = rules_df.copy()
    rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
    
    for col in ['support', 'confidence', 'lift', 'leverage', 'conviction']:
        if col in rules_df.columns:
            rules_df[col] = rules_df[col].apply(lambda x: f"{x:.4f}")
    
    html = '<table class="table table-striped table-hover">'
    html += '<thead><tr>'
    html += '<th>Rule</th>'
    html += f'<th>{metric_name.capitalize()}</th>'
    html += '<th>Support</th>'
    html += '<th>Confidence</th>'
    html += '<th>Lift</th>'
    html += '</tr></thead>'
    html += '<tbody>'
    
    for idx, row in rules_df.iterrows():
        html += '<tr>'
        html += f'<td>{row["antecedents"]} =&gt; {row["consequents"]}</td>' 
        html += f'<td>{row[metric_name]}</td>'
        html += f'<td>{row["support"]}</td>'
        html += f'<td>{row["confidence"]}</td>'
        html += f'<td>{row["lift"]}</td>'
        html += '</tr>'
    
    html += '</tbody></table>'
    return html

def create_network_visualization(rules_df, output_path, title):
    """Create a network visualization of association rules"""
    plt.figure(figsize=(12, 8))
    
    G = nx.DiGraph()
    
    for idx, row in rules_df.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        for a in antecedents:
            for c in consequents:
                G.add_node(a)
                G.add_node(c)
                G.add_edge(a, c, weight=row['lift'])
    
    try:
        pos = nx.spring_layout(G, k=0.15, iterations=20)
    except:
        pos = nx.spring_layout(G)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
        norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
        cmap = plt.cm.viridis
        
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='skyblue', 
                node_size=1500, 
                font_size=8,
                font_weight='bold', 
                edge_color=[cmap(norm(w)) for w in weights],
                width=2.0,
                alpha=0.7,
                arrows=True, 
                arrowsize=15)
    else:
        nx.draw(G, pos, with_labels=True, node_color='skyblue')
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_matrix_visualization(rules_df, output_path):
    """Create a matrix visualization of association rules"""
    plt.figure(figsize=(12, 10))
    
    all_items = set()
    for idx, row in rules_df.iterrows():
        all_items.update(row['antecedents'])
        all_items.update(row['consequents'])
    
    all_items = list(all_items)
    n_items = len(all_items)
    
    matrix = np.zeros((n_items, n_items))
    
    for idx, row in rules_df.iterrows():
        for a in row['antecedents']:
            for c in row['consequents']:
                i = all_items.index(a)
                j = all_items.index(c)
                matrix[i, j] = row['lift']
    
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='Lift')
    
    plt.xticks(range(n_items), all_items, rotation=90, fontsize=8)
    plt.yticks(range(n_items), all_items, fontsize=8)
    
    plt.title('Association Rules Matrix (Lift Values)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_parallel_coordinates(rules_df, output_path):
    """Create a parallel coordinates plot of association rules"""
    plt.figure(figsize=(15, 8))
    
    top_n = min(len(rules_df), 10) 
    top_rules = rules_df.head(top_n)
    
    for idx, rule in enumerate(range(top_n)):
        row = top_rules.iloc[rule]
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        for a in antecedents:
            for c in consequents:
                a_short = a.split('_')[-1] if '_' in a else a
                c_short = c.split('_')[-1] if '_' in c else c
                
                plt.plot([1, 2], [idx, idx], 'o-', 
                         linewidth=2, 
                         alpha=0.7,
                         color=plt.cm.viridis(idx/top_n))
                
                plt.text(1, idx, a_short, ha='right', va='center', fontsize=9)
                plt.text(2, idx, c_short, ha='left', va='center', fontsize=9)
    
    plt.xticks([1, 2], ['Antecedents', 'Consequents'], fontsize=12)
    plt.yticks([])
    plt.title('Association Rules Parallel Coordinates', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_arm_analysis(df, output_dir='static/arm_results'):
    """Main function to run the entire ARM analysis pipeline"""
    data_prep_results = prepare_flight_data_for_arm(df)
    
    arm_results = perform_arm_analysis(data_prep_results['transactions'], output_dir)
    
    arm_results.update({
        'sample_transactions': data_prep_results['sample_transactions'],
        'num_transactions': data_prep_results['num_transactions']
    })
    
    return arm_results

if __name__ == "__main__":
    df = pd.read_csv('flight_data.csv')
    
    results = run_arm_analysis(df)
    
    print("ARM Analysis Results:")
    print(f"Total transactions: {results['num_transactions']}")
    print(f"Rules found (by support): {results['num_rules_support']}")
    print(f"Rules found (by confidence): {results['num_rules_confidence']}")
    print(f"Rules found (by lift): {results['num_rules_lift']}")