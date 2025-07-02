import pandas as pd
import numpy as np
import logging

def get_overview(csv_path):
    """
    Generate overview statistics for the cleaned dataset
    
    Args:
        csv_path: Path to the cleaned CSV file
    
    Returns:
        dict: Overview data including stats, column info, and summary
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Basic info
        overview = {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'columns': [],
            'summary_stats': {},
            'data_types': {},
            'missing_values': {}
        }
        
        # Column information
        for col in df.columns:
            col_info = {
                'name': col,
                'type': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'sample_values': df[col].dropna().head(5).tolist()
            }
            
            # Add specific stats for numeric columns
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()) if df[col].std() is not pd.NaType else 0
                })
            
            overview['columns'].append(col_info)
        
        # Summary statistics for numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            overview['summary_stats'] = numeric_df.describe().to_dict()
        
        # Data types summary
        overview['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Missing values summary
        overview['missing_values'] = df.isnull().sum().to_dict()
        
        # Additional insights
        overview['insights'] = {
            'total_cells': len(df) * len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        return overview
        
    except Exception as e:
        logging.error(f"Error generating overview: {str(e)}")
        raise e
