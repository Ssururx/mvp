from flask import Flask, render_template
import pandas as pd
import numpy as np
from scipy import stats

app = Flask(__name__)

# Sample data - would normally load from file/database
SAMPLE_DATA = {
    'home_value': [350000, 400000, 450000, 500000, 550000],
    'down_payment': [70000, 80000, 90000, 100000, 110000],
    'loan_amount': [280000, 320000, 360000, 400000, 440000],
    'interest_rate': [3.5, 3.8, 4.0, 4.2, 4.5],
    'monthly_payment': [1260, 1370, 1500, 1590, 1690]
}

def calculate_general_stats(df):
    """Calculate general statistics that work for any dataset"""
    stats = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) >= 4:
        stats['stat1_value'] = f"${df[numeric_cols[0]].mean():,.0f}"
        stats['stat1_label'] = f"Avg {numeric_cols[0].replace('_', ' ').title()}"
        stats['stat2_value'] = f"{df[numeric_cols[1]].mean():.2f}%"
        stats['stat2_label'] = f"Avg {numeric_cols[1].replace('_', ' ').title()}"
        stats['stat3_value'] = f"${df[numeric_cols[2]].mean():,.0f}"
        stats['stat3_label'] = f"Avg {numeric_cols[2].replace('_', ' ').title()}"
        stats['stat4_value'] = f"{df[numeric_cols[3]].mean():.0f}%"
        stats['stat4_label'] = f"Avg {numeric_cols[3].replace('_', ' ').title()}"
    else:
        stats['stat1_value'] = f"{len(df):,}"
        stats['stat1_label'] = "Total Records"
        stats['stat2_value'] = f"{df.shape[1]}"
        stats['stat2_label'] = "Number of Features"
        stats['stat3_value'] = f"{df.isna().sum().sum()}"
        stats['stat3_label'] = "Missing Values"
        stats['stat4_value'] = f"{df.duplicated().sum()}"
        stats['stat4_label'] = "Duplicate Rows"
    
    return stats

def prepare_data_sample(df, sample_size=5):
    """Prepare data sample for display with proper formatting"""
    sample_df = df.head(sample_size)
    formatted_data = []
    
    for _, row in sample_df.iterrows():
        formatted_row = {}
        for col, value in row.items():
            if pd.api.types.is_numeric_dtype(df[col]):
                if 'interest' in col.lower() or 'rate' in col.lower():
                    formatted_row[col] = f"{value:.2f}%"
                elif 'value' in col.lower() or 'amount' in col.lower() or 'payment' in col.lower():
                    formatted_row[col] = f"${value:,.0f}"
                else:
                    formatted_row[col] = f"{value:,.2f}"
            else:
                formatted_row[col] = str(value)
        formatted_data.append(formatted_row)
    
    return {
        'columns': [col.replace('_', ' ').title() for col in df.columns],
        'sample_data': formatted_data,
        'total_records': len(df)
    }

def calculate_correlation_matrix(df):
    """Calculate and format correlation matrix"""
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr().round(2)
    columns = [col.replace('_', ' ').title() for col in corr_matrix.columns]
    rows = []
    
    for i, row in enumerate(corr_matrix.itertuples()):
        row_data = {'variable': columns[i], 'values': []}
        for j, value in enumerate(row[1:]):
            cell = {'value': f"{value:.2f}", 'class': ''}
            if i != j:  # Skip diagonal
                if value > 0.7:
                    cell['class'] = 'positive-correlation'
                elif value < -0.7:
                    cell['class'] = 'negative-correlation'
            row_data['values'].append(cell)
        rows.append(row_data)
    
    return {'columns': columns, 'rows': rows}

def generate_correlation_insights(df):
    """Generate human-readable insights from correlations"""
    insights = []
    numeric_df = df.select_dtypes(include=['number'])
    
    if len(numeric_df.columns) < 2:
        return insights
    
    corr_matrix = numeric_df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    
    top_corrs = (corr_matrix.unstack()
                 .sort_values(ascending=False)
                 .drop_duplicates()
                 .head(3))
    
    for (col1, col2), corr_value in top_corrs.items():
        if pd.isna(corr_value) or corr_value < 0.3:
            continue
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            numeric_df[col1], numeric_df[col2])
        
        insight = {
            'title': f"{col1.replace('_', ' ').title()} ↔ {col2.replace('_', ' ').title()} (r={corr_value:.2f})",
            'text': generate_insight_text(col1, col2, corr_value, slope),
            'tags': generate_insight_tags(corr_value, p_value),
            'icon': get_insight_icon(col1, col2)
        }
        insights.append(insight)
    
    return insights

def generate_insight_text(col1, col2, corr_value, slope):
    """Generate text explanation for correlation"""
    col1_clean = col1.replace('_', ' ').lower()
    col2_clean = col2.replace('_', ' ').lower()
    
    strength = "Nearly perfect" if corr_value > 0.9 else \
               "Strong" if corr_value > 0.7 else \
               "Moderate" if corr_value > 0.5 else "Weak"
    
    direction = "positive" if slope > 0 else "negative"
    
    if 'price' in col1_clean and 'size' in col2_clean:
        return (f"{strength} {direction} correlation between {col1_clean} and {col2_clean}. "
                f"For each unit increase in {col1_clean}, {col2_clean} changes by {abs(slope):.2f} units.")
    elif 'time' in col1_clean and 'score' in col2_clean:
        return (f"{strength} {direction} relationship between {col1_clean} and {col2_clean}. "
                f"Longer {col1_clean} is associated with {'higher' if direction == 'positive' else 'lower'} {col2_clean}.")
    
    return (f"{strength} {direction} correlation between {col1_clean} and {col2_clean}. "
            f"The relationship suggests that as {col1_clean} increases, {col2_clean} tends to "
            f"{'increase' if direction == 'positive' else 'decrease'}.")

def generate_insight_tags(corr_value, p_value):
    """Generate tags for correlation insights"""
    tags = []
    
    if corr_value > 0.7:
        tags.append("Strong")
    elif corr_value > 0.3:
        tags.append("Moderate")
    else:
        tags.append("Weak")
    
    if p_value < 0.01:
        tags.append("Highly Significant")
    elif p_value < 0.05:
        tags.append("Significant")
    
    tags.append("Linear" if abs(corr_value) > 0.9 else "Non-linear")
    
    return tags[:3]

def get_insight_icon(col1, col2):
    """Get appropriate icon based on column names"""
    col1_lower = col1.lower()
    col2_lower = col2.lower()
    
    money_terms = {'price', 'cost', 'value', 'amount', 'payment', 'income', 'revenue'}
    time_terms = {'time', 'date', 'year', 'month', 'day', 'hour'}
    
    if any(term in col1_lower or term in col2_lower for term in money_terms):
        return "fas fa-dollar-sign"
    elif any(term in col1_lower or term in col2_lower for term in time_terms):
        return "fas fa-clock"
    elif 'age' in col1_lower or 'age' in col2_lower:
        return "fas fa-birthday-cake"
    return "fas fa-chart-line"

@app.route('/')
def dashboard():
    df = pd.DataFrame(SAMPLE_DATA)
    
    context = {
        'stats': calculate_general_stats(df),
        'data_sample': prepare_data_sample(df),
        'correlation_data': calculate_correlation_matrix(df),
        'correlation_insights': generate_correlation_insights(df)
    }
    
    return render_template('dashboard.html', **context)

if __name__ == '__main__':
    app.run(debug=True)
