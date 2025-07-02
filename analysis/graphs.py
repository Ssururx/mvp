import pandas as pd
import numpy as np
import logging

def generate_graph_data(csv_path, x_column, y_column):
    """
    Generate data for plotting X-Y graphs
    
    Args:
        csv_path: Path to the cleaned CSV file
        x_column: Name of the X-axis column
        y_column: Name of the Y-axis column
    
    Returns:
        dict: Graph data with x, y values and metadata
    """
    try:
        df = pd.read_csv(csv_path)
        
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in dataset")
        
        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in dataset")
        
        # Get the data for plotting
        x_data = df[x_column].dropna()
        y_data = df[y_column].dropna()
        
        # Ensure we have the same number of points
        min_length = min(len(x_data), len(y_data))
        x_data = x_data.head(min_length)
        y_data = y_data.head(min_length)
        
        # Convert to appropriate types for JSON serialization
        x_values = x_data.tolist()
        y_values = y_data.tolist()
        
        # Determine chart type based on data types
        chart_type = "scatter"
        if df[x_column].dtype == 'object' or df[y_column].dtype == 'object':
            chart_type = "bar"
        
        # Calculate correlation if both are numeric
        correlation = None
        if df[x_column].dtype in ['int64', 'float64'] and df[y_column].dtype in ['int64', 'float64']:
            correlation = float(df[x_column].corr(df[y_column]))
        
        graph_data = {
            'x_values': x_values,
            'y_values': y_values,
            'x_label': x_column,
            'y_label': y_column,
            'chart_type': chart_type,
            'data_points': len(x_values),
            'correlation': correlation,
            'x_type': str(df[x_column].dtype),
            'y_type': str(df[y_column].dtype)
        }
        
        # Add statistical summary
        if df[x_column].dtype in ['int64', 'float64']:
            graph_data['x_stats'] = {
                'min': float(df[x_column].min()),
                'max': float(df[x_column].max()),
                'mean': float(df[x_column].mean()),
                'std': float(df[x_column].std()) if pd.notna(df[x_column].std()) else 0
            }
        
        if df[y_column].dtype in ['int64', 'float64']:
            graph_data['y_stats'] = {
                'min': float(df[y_column].min()),
                'max': float(df[y_column].max()),
                'mean': float(df[y_column].mean()),
                'std': float(df[y_column].std()) if pd.notna(df[y_column].std()) else 0
            }
        
        return graph_data
        
    except Exception as e:
        logging.error(f"Error generating graph data: {str(e)}")
        raise e
