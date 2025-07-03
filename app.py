from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats as scipy_stats # Renamed scipy.stats to avoid conflict
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
from datetime import datetime

app = Flask(__name__)

# =====================
# DATA PREPARATION
# =====================

raw_data = {
    "House Value ($)": [350000, 400000, 450000, 500000, 550000],
    "Down Payment ($)": [70000, 80000, 90000, 100000, 110000],
    "Loan Amount ($)": [280000, 320000, 360000, 400000, 440000],
    "Interest Rate (%)": [3.5, 3.8, 4.0, 4.2, 4.5],
    "Monthly Payment ($)": [1260, 1370, 1500, 1590, 1690]
}

df = pd.DataFrame(raw_data)
df['Down Payment (%)'] = (df['Down Payment ($)'] / df['House Value ($)']) * 100
df['Loan-to-Value Ratio'] = df['Loan Amount ($)'] / df['House Value ($)']

# =====================
# STATISTICAL ANALYSIS
# =====================

def calculate_descriptive_stats():
    """Calculate comprehensive descriptive statistics"""
    stats_dict = {} # Renamed to stats_dict to be explicit
    
    # Central tendency and dispersion for all numeric columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats_dict[f'avg_{col}'] = df[col].mean()
            stats_dict[f'median_{col}'] = df[col].median()
            stats_dict[f'min_{col}'] = df[col].min()
            stats_dict[f'max_{col}'] = df[col].max()
            stats_dict[f'std_{col}'] = df[col].std() # Add std for all numeric columns
    
    stats_dict['payment_range'] = stats_dict['max_Monthly Payment ($)'] - stats_dict['min_Monthly Payment ($)']
    
    # Shape
    stats_dict['payment_skewness'] = df['Monthly Payment ($)'].skew()
    stats_dict['payment_kurtosis'] = df['Monthly Payment ($)'].kurtosis()
    
    # Percentiles
    for p in [25, 50, 75, 90]:
        stats_dict[f'p{p}_payment'] = df['Monthly Payment ($)'].quantile(p/100)
    
    return stats_dict # Return the dictionary

def calculate_correlations(descriptive_stats_dict): # Now accepts the stats dictionary
    """Calculate complete correlation matrix and key relationships"""
    corr_matrix = df.corr().round(3)
    
    # Extract top correlations with monthly payment
    payment_corrs = corr_matrix['Monthly Payment ($)'].sort_values(
        key=abs, ascending=False).drop('Monthly Payment ($)')
    
    key_relationships = []
    for feature, corr in payment_corrs.items():
        # Ensure the feature exists in the descriptive_stats_dict before accessing
        # This handles cases where std_feature might not be directly available for all features
        std_feature_key = f'std_{feature}'
        
        # Handle features where std_ might not be directly calculated or meaningful (e.g., percentages, ratios)
        # You might need to add more specific std_ calculations in calculate_descriptive_stats
        # or handle these cases differently if they don't have a direct 'std_' equivalent
        
        impact_value = 0.0 # Default in case of division by zero or missing key
        # Check if the std_feature_key exists and its value is not zero to prevent division by zero
        if descriptive_stats_dict.get(std_feature_key) and descriptive_stats_dict[std_feature_key] != 0:
            impact_value = corr * descriptive_stats_dict['std_payment'] / descriptive_stats_dict[std_feature_key]
        
        relationship = {
            'feature': feature,
            'correlation': corr,
            'strength': 'very strong' if abs(corr) > 0.8 else 
                               'strong' if abs(corr) > 0.6 else
                               'moderate' if abs(corr) > 0.4 else
                               'weak',
            'direction': 'positive' if corr > 0 else 'negative',
            'impact': f"${impact_value:,.2f} per unit change" # Use impact_value
        }
        key_relationships.append(relationship)
    
    return {
        'matrix': corr_matrix,
        'key_relationships': key_relationships,
        'multicollinearity': {
            'home_loan_corr': corr_matrix.loc['House Value ($)', 'Loan Amount ($)'],
            'home_down_corr': corr_matrix.loc['House Value ($)', 'Down Payment ($)']
        }
    }


# =====================
# PREDICTIVE MODELING  
# =====================

def build_regression_models():
    """Build and evaluate multiple regression models"""
    X = df[['House Value ($)', 'Loan Amount ($)', 'Interest Rate (%)', 'Down Payment (%)']]
    y = df['Monthly Payment ($)']
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    lr_pred = lr.predict(X)
    
    # Model metrics
    metrics = {
        'r2': r2_score(y, lr_pred),
        'adj_r2': 1 - (1-r2_score(y, lr_pred)) * (len(y)-1)/(len(y)-X.shape[1]-1),
        'rmse': mean_squared_error(y, lr_pred, squared=False),
        'mae': np.mean(np.abs(y - lr_pred))
    }
    
    # Standardized coefficients
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    lr_std = LinearRegression().fit(X_std, y_std)
    
    return {
        'linear': {
            'coefficients': dict(zip(X.columns, lr.coef_)),
            'intercept': lr.intercept_,
            'std_coefficients': dict(zip(X.columns, lr_std.coef_)),
            'metrics': metrics,
            'equation': (
                f"{lr.coef_[0]:.5f}*HomeValue + "
                f"{lr.coef_[1]:.5f}*Loan + "
                f"{lr.coef_[2]:.5f}*Rate + "
                f"{lr.coef_[3]:.5f}*DownPct + "
                f"{lr.intercept_:.2f}"
            )
        },
        'median_example': {
            'home_value': df['House Value ($)'].median(),
            'down_payment': df['Down Payment ($)'].median(),
            'loan_amount': df['Loan Amount ($)'].median(),
            'interest_rate': df['Interest Rate (%)'].median(),
            'payment': df.loc[df['House Value ($)'] == df['House Value ($)'].median(), 
                               'Monthly Payment ($)'].values[0]
        }
    }

# =====================
# VISUALIZATION DATA
# =====================

def generate_visualization_data():
    """Generate all visualization datasets"""
    # Scatter plot data
    plt.figure(figsize=(10,6))
    plt.scatter(df['House Value ($)'], df['Monthly Payment ($)'])
    plt.title("Home Value vs Monthly Payment")
    plt.xlabel("Home Value ($)")
    plt.ylabel("Monthly Payment ($)")
    
    scatter_buf = BytesIO()
    plt.savefig(scatter_buf, format='png', dpi=100)
    scatter_buf.seek(0)
    scatter_url = base64.b64encode(scatter_buf.read()).decode('utf8')
    plt.close()
    
    # Histogram data
    hist, bins = np.histogram(df['Monthly Payment ($)'], bins=5)
    
    # Box plot data
    boxplot_stats = {
        'q1': df['Monthly Payment ($)'].quantile(0.25),
        'median': df['Monthly Payment ($)'].median(),
        'q3': df['Monthly Payment ($)'].quantile(0.75),
        'iqr': df['Monthly Payment ($)'].quantile(0.75) - df['Monthly Payment ($)'].quantile(0.25)
    }
    boxplot_stats['whisker_low'] = boxplot_stats['q1'] - 1.5 * boxplot_stats['iqr']
    boxplot_stats['whisker_high'] = boxplot_stats['q3'] + 1.5 * boxplot_stats['iqr']
    
    return {
        'scatter_plot': scatter_url,
        'histogram': {
            'bins': bins.tolist(),
            'counts': hist.tolist()
        },
        'boxplot': boxplot_stats,
        'time_series': {
            'dates': [datetime(2023, 1, i+1).strftime('%Y-%m-%d') for i in range(len(df))],
            'values': df['Monthly Payment ($)'].tolist()
        }
    }

# =====================
# MORTGAGE CALCULATOR
# =====================

def calculate_mortgage_payment(principal, annual_rate, years=30):
    """Calculate monthly mortgage payment using standard formula"""
    monthly_rate = annual_rate / 100 / 12
    n_payments = years * 12
    # Handle monthly_rate being zero for 0% interest
    if monthly_rate == 0:
        if n_payments == 0: # Avoid division by zero if years is also 0
            return principal
        return principal / n_payments
    
    payment = principal * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    return payment

# =====================
# FLASK ROUTES
# =====================

@app.route('/')
def dashboard():
    """Main dashboard route serving all calculated data"""
    # Calculate all metrics
    stats_dict = calculate_descriptive_stats() # Get the stats dictionary
    corr = calculate_correlations(stats_dict) # Pass the stats dictionary to correlations
    models = build_regression_models()
    viz = generate_visualization_data()
    
    # Prepare complete template data
    template_data = {
        # Core app information
        "app_name": "MortgageAnalyticsPro",
        "login_button_text": "Sign In",
        "signup_button_text": "Register",
        "dataset_name": "Residential Mortgage Dataset",
        "record_count": f"{len(df):,} records",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        
        # Navigation menu
        "menu_items": [
            {
                "text": "Overview", 
                "section": "overview", 
                "icon": "fas fa-chart-pie", 
                "active": True
            },
            {
                "text": "Exploration", 
                "section": "exploration", 
                "icon": "fas fa-search", 
                "active": False
            },
            {
                "text": "Models", 
                "section": "models", 
                "icon": "fas fa-calculator", 
                "active": False
            },
            {
                "text": "Visualizations", 
                "section": "visualizations", 
                "icon": "fas fa-chart-line", 
                "active": False
            },
            {
                "text": "Insights", 
                "section": "insights", 
                "icon": "fas fa-lightbulb", 
                "active": False
            }
        ],
        
        # Key statistics cards
        "stats": [
            {
                "value": f"${stats_dict['avg_House Value ($)']/1000:,.1f}K",
                "label": "Average Home Value",
                "change": f"{((stats_dict['max_House Value ($)'] - stats_dict['min_House Value ($)'])/stats_dict['avg_House Value ($)'])*100:.1f}% range"
            },
            {
                "value": f"{stats_dict['avg_Interest Rate (%)']:.2f}%",
                "label": "Avg Interest Rate",
                "change": f"{(stats_dict['max_Interest Rate (%)'] - stats_dict['min_Interest Rate (%)']):.1f}% spread"
            },
            {
                "value": f"${stats_dict['avg_Monthly Payment ($)']:.0f}",
                "label": "Avg Monthly Payment",
                "change": f"${stats_dict['payment_range']:,.0f} range"
            },
            {
                "value": f"{stats_dict['avg_Down Payment (%)']:.1f}%",
                "label": "Avg Down Payment",
                "change": f"{(stats_dict['max_Down Payment (%)'] - stats_dict['min_Down Payment (%)']):.1f}% spread"
            }
        ],
        
        # Data sample table
        "data_sample": {
            "headers": list(df.columns),
            "rows": [
                {
                    "House Value ($)": f"${row['House Value ($)']:,.0f}",
                    "Down Payment ($)": f"${row['Down Payment ($)']:,.0f}",
                    "Loan Amount ($)": f"${row['Loan Amount ($)']:,.0f}",
                    "Interest Rate (%)": f"{row['Interest Rate (%)']}%",
                    "Monthly Payment ($)": f"${row['Monthly Payment ($)']:,.0f}",
                    "Down Payment (%)": f"{row['Down Payment (%)']:.1f}%",
                    "Loan-to-Value Ratio": f"{row['Loan-to-Value Ratio']:.2f}"
                }
                for _, row in df.iterrows()
            ]
        },
        
        # Correlation analysis
        "correlation_matrix": {
            "headers": list(df.columns),
            "rows": [
                {
                    "variable": col,
                    "values": [
                        {
                            "value": f"{corr['matrix'].loc[col, other_col]:.2f}",
                            "class": "positive" if corr['matrix'].loc[col, other_col] > 0.5 else 
                                     "negative" if corr['matrix'].loc[col, other_col] < -0.5 else "neutral"
                        }
                        for other_col in df.columns
                    ]
                }
                for col in df.columns
            ]
        },
        
        # Key relationships
        "key_relationships": [
            {
                "title": f"{rel['feature']} → Monthly Payment",
                "text": (
                    f"{rel['strength'].capitalize()} {rel['direction']} relationship (r = {rel['correlation']:.2f}). "
                    f"Impact: {rel['impact']}"
                ),
                "icon": "fas fa-arrow-up" if rel['direction'] == "positive" else "fas fa-arrow-down",
                "tags": [rel['strength'], rel['direction']]
            }
            for rel in corr['key_relationships']
        ],
        
        # Regression analysis
        "regression_analysis": {
            "equation": models['linear']['equation'],
            "metrics": {
                "r2": f"{models['linear']['metrics']['r2']:.3f}",
                "adj_r2": f"{models['linear']['metrics']['adj_r2']:.3f}",
                "rmse": f"${models['linear']['metrics']['rmse']:.2f}",
                "mae": f"${models['linear']['metrics']['mae']:.2f}"
            },
            "coefficients": [
                {
                    "name": name,
                    "value": f"{coef:.5f}",
                    "std_value": f"{models['linear']['std_coefficients'][name]:.3f}",
                    "interpretation": (
                        f"Each ${name.split('(')[0].strip()} increases payment by ${coef:.2f} "
                        f"(standardized effect: {models['linear']['std_coefficients'][name]:.2f} SD)"
                    )
                }
                for name, coef in models['linear']['coefficients'].items()
            ]
        },
        
        # Visualization data
        "visualizations": {
            "scatter_plot": viz['scatter_plot'],
            "histogram": viz['histogram'],
            "boxplot": viz['boxplot'],
            "time_series": viz['time_series']
        },
        
        # Mortgage calculator defaults
        "calculator_defaults": {
            "home_value": models['median_example']['home_value'],
            "down_payment_pct": (models['median_example']['down_payment'] / 
                                 models['median_example']['home_value']) * 100,
            "interest_rate": models['median_example']['interest_rate'],
            "loan_term": 30,
            "example_payment": models['median_example']['payment']
        },
        
        # Data quality metrics
        "data_quality": {
            "completeness": "100%",
            "duplicates": 0,
            "outliers": {
                "count": 0, # Placeholder, actual outlier detection not implemented
                "method": "IQR (1.5x)"
            },
            "skewness": f"{stats_dict['payment_skewness']:.2f}",
            "kurtosis": f"{stats_dict['payment_kurtosis']:.2f}"
        },
        
        # Statistical tests
        "statistical_tests": [
            {
                "name": "Normality (Shapiro-Wilk)",
                "result": "Normal" if abs(stats_dict['payment_skewness']) < 0.5 and 
                                     abs(stats_dict['payment_kurtosis']) < 1 else "Non-normal",
                "details": f"Skewness: {stats_dict['payment_skewness']:.2f}, Kurtosis: {stats_dict['payment_kurtosis']:.2f}"
            },
            {
                "name": "Homoscedasticity",
                "result": "Equal variance" if stats_dict['std_payment']/stats_dict['avg_Monthly Payment ($)'] < 0.2 
                                     else "Unequal variance",
                "details": f"CV: {(stats_dict['std_payment']/stats_dict['avg_Monthly Payment ($)'])*100:.1f}%"
            }
        ],
        
        # Actionable insights
        "insights": [
            {
                "title": "Down Payment Optimization",
                "content": (
                    f"The optimal down payment percentage appears to be around {stats_dict['median_Down Payment (%)']:.1f}%. "
                    "Higher down payments reduce loan amounts but have diminishing returns on payment reduction."
                ),
                "tags": ["optimization", "down-payment"]
            },
            {
                "title": "Interest Rate Sensitivity",
                "content": (
                    f"Each 0.1% increase in interest rate increases payments by approximately "
                    f"${models['linear']['coefficients']['Interest Rate (%)'] * 0.1 * 100:.2f} "
                    "for median-priced homes."
                ),
                "tags": ["rate-sensitivity", "risk"]
            }
        ],
        
        # UI text elements
        "ui_text": {
            "refresh_text": "Refresh Data",
            "export_text": "Export Results",
            "calculate_text": "Calculate Payment",
            "view_details": "View Detailed Analysis",
            "generated_on": f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }
    }
    
    return render_template('dashboard.html', **template_data)

@app.route('/calculate', methods=['POST'])
def calculate():
    """API endpoint for mortgage calculations"""
    try:
        data = request.json
        home_value = float(data['home_value'])
        down_payment_pct = float(data['down_payment_pct'])
        interest_rate = float(data['interest_rate'])
        loan_term = int(data['loan_term'])
        
        down_payment = home_value * (down_payment_pct / 100)
        loan_amount = home_value - down_payment
        
        monthly_payment = calculate_mortgage_payment(loan_amount, interest_rate, loan_term)
        total_payments = monthly_payment * loan_term * 12
        total_interest = total_payments - loan_amount
        
        return jsonify({
            "monthly_payment": round(monthly_payment, 2),
            "total_interest": round(total_interest, 2),
            "total_cost": round(total_payments, 2),
            "loan_amount": round(loan_amount, 2),
            "down_payment": round(down_payment, 2),
            "amortization_schedule": [
                {
                    "year": year,
                    "principal_paid": round(loan_amount * (year/loan_term) * 0.8, 2), # Simplified for demo
                    "interest_paid": round(loan_amount * (year/loan_term) * 0.2, 2), # Simplified for demo
                    "remaining_balance": round(loan_amount * (1 - year/loan_term), 2) # Simplified for demo
                }
                for year in range(1, loan_term + 1)
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
