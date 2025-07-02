import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import sympy as sp
import logging

def find_equation(csv_path, x_column, y_column):
    """
    Find the best-fit equation for X-Y data using symbolic regression
    
    Args:
        csv_path: Path to the cleaned CSV file
        x_column: Name of the X-axis column
        y_column: Name of the Y-axis column
    
    Returns:
        dict: Equation data with formula, R-squared, and parameters
    """
    try:
        df = pd.read_csv(csv_path)
        
        if x_column not in df.columns:
            raise ValueError(f"Column '{x_column}' not found in dataset")
        
        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in dataset")
        
        # Ensure both columns are numeric
        if df[x_column].dtype not in ['int64', 'float64']:
            raise ValueError(f"Column '{x_column}' must be numeric for equation fitting")
        
        if df[y_column].dtype not in ['int64', 'float64']:
            raise ValueError(f"Column '{y_column}' must be numeric for equation fitting")
        
        # Get clean data
        clean_df = df[[x_column, y_column]].dropna()
        
        if len(clean_df) < 3:
            raise ValueError("Not enough data points for equation fitting")
        
        X = clean_df[x_column].values.reshape(-1, 1)
        y = clean_df[y_column].values
        
        equations = []
        
        # Try linear regression
        try:
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            y_pred = linear_model.predict(X)
            r2_linear = r2_score(y, y_pred)
            
            # Create symbolic equation
            x_sym = sp.Symbol('x')
            linear_eq = linear_model.coef_[0] * x_sym + linear_model.intercept_
            
            equations.append({
                'type': 'linear',
                'formula': str(linear_eq),
                'r_squared': float(r2_linear),
                'coefficients': {
                    'slope': float(linear_model.coef_[0]),
                    'intercept': float(linear_model.intercept_)
                }
            })
        except Exception as e:
            logging.warning(f"Linear regression failed: {str(e)}")
        
        # Try polynomial regression (degree 2)
        try:
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            poly_model = LinearRegression()
            poly_model.fit(X_poly, y)
            y_pred_poly = poly_model.predict(X_poly)
            r2_poly = r2_score(y, y_pred_poly)
            
            # Create symbolic equation
            x_sym = sp.Symbol('x')
            poly_eq = (poly_model.coef_[2] * x_sym**2 + 
                      poly_model.coef_[1] * x_sym + 
                      poly_model.intercept_)
            
            equations.append({
                'type': 'quadratic',
                'formula': str(poly_eq),
                'r_squared': float(r2_poly),
                'coefficients': {
                    'a': float(poly_model.coef_[2]),
                    'b': float(poly_model.coef_[1]),
                    'c': float(poly_model.intercept_)
                }
            })
        except Exception as e:
            logging.warning(f"Polynomial regression failed: {str(e)}")
        
        # Try exponential fit (if all y values are positive)
        if all(y > 0):
            try:
                log_y = np.log(y)
                exp_model = LinearRegression()
                exp_model.fit(X, log_y)
                y_pred_exp = np.exp(exp_model.predict(X))
                r2_exp = r2_score(y, y_pred_exp)
                
                # Create symbolic equation
                x_sym = sp.Symbol('x')
                exp_eq = sp.exp(exp_model.coef_[0] * x_sym + exp_model.intercept_)
                
                equations.append({
                    'type': 'exponential',
                    'formula': str(exp_eq),
                    'r_squared': float(r2_exp),
                    'coefficients': {
                        'a': float(np.exp(exp_model.intercept_)),
                        'b': float(exp_model.coef_[0])
                    }
                })
            except Exception as e:
                logging.warning(f"Exponential regression failed: {str(e)}")
        
        if not equations:
            raise ValueError("No valid equations could be fitted to the data")
        
        # Find the best equation based on R-squared
        best_equation = max(equations, key=lambda eq: eq['r_squared'])
        
        result = {
            'best_equation': best_equation,
            'all_equations': equations,
            'data_points': len(clean_df),
            'x_column': x_column,
            'y_column': y_column
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Error finding equation: {str(e)}")
        raise e
