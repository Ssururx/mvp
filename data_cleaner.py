import os
import pandas as pd
import numpy as np
import logging
from app import app

def clean_data(filepath, file_id):
    """
    Clean the uploaded CSV data and save to cleaned directory
    
    Args:
        filepath: Path to the uploaded CSV file
        file_id: Database ID of the uploaded file
    
    Returns:
        str: Filename of the cleaned CSV
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        logging.info(f"Original data shape: {df.shape}")
        
        # Basic cleaning steps
        original_rows = len(df)
        
        # 1. Remove completely empty rows
        df = df.dropna(how='all')
        
        # 2. Remove duplicate rows
        df = df.drop_duplicates()
        
        # 3. Handle missing values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Forward fill missing values
            df[col] = df[col].fillna(method='ffill')
            # If still missing values, fill with median
            df[col] = df[col].fillna(df[col].median())
        
        # 4. Handle missing values in categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            # Forward fill missing values
            df[col] = df[col].fillna(method='ffill')
            # If still missing values, fill with mode or 'Unknown'
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna('Unknown')
        
        # 5. Convert data types appropriately
        for col in df.columns:
            # Try to convert to numeric if possible
            if df[col].dtype == 'object':
                try:
                    # Check if it's a number stored as string
                    pd.to_numeric(df[col], errors='raise')
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    pass
        
        # 6. Remove any remaining rows with NaN values
        df = df.dropna()
        
        cleaned_rows = len(df)
        logging.info(f"Cleaned data shape: {df.shape}")
        logging.info(f"Removed {original_rows - cleaned_rows} rows during cleaning")
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        # Save cleaned data
        cleaned_filename = f"cleaned_{file_id}.csv"
        cleaned_path = os.path.join(app.config['CLEANED_FOLDER'], cleaned_filename)
        df.to_csv(cleaned_path, index=False)
        
        logging.info(f"Cleaned data saved to: {cleaned_path}")
        return cleaned_filename
        
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise e
