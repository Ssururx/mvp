import os
import json
import pandas as pd
import numpy as np
import logging
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI client only if API key is available
openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logging.warning(f"Failed to initialize OpenAI client: {str(e)}")
        openai_client = None

def generate_insights(csv_path):
    """
    Generate AI-powered insights and analysis of the dataset
    
    Args:
        csv_path: Path to the cleaned CSV file
    
    Returns:
        dict: AI-generated insights and analysis
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Prepare data summary for AI analysis
        data_summary = {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'columns': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns)
        }
        
        # Add statistical summary for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            data_summary['statistics'] = numeric_df.describe().to_dict()
        
        # Calculate correlations between numeric columns
        if len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()
            # Get strongest correlations (excluding self-correlations)
            correlations = []
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicates and self-correlations
                        corr_value = correlation_matrix.loc[col1, col2]
                        if not pd.isna(corr_value):
                            correlations.append({
                                'column1': col1,
                                'column2': col2,
                                'correlation': float(corr_value)
                            })
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            data_summary['top_correlations'] = correlations[:5]  # Top 5 correlations
        
        # Sample data for context
        data_summary['sample_data'] = df.head(5).to_dict('records')
        
        # Create prompt for AI analysis
        prompt = f"""
        Analyze this dataset and provide comprehensive insights. The data summary is:
        
        {json.dumps(data_summary, indent=2)}
        
        Please provide a detailed analysis including:
        1. Key patterns and trends in the data
        2. Notable correlations and relationships
        3. Data quality observations
        4. Potential business or research insights
        5. Recommendations for further analysis
        6. Any anomalies or interesting findings
        
        Format your response as JSON with the following structure:
        {{
            "summary": "Brief overview of the dataset",
            "key_findings": ["finding1", "finding2", ...],
            "correlations_analysis": "Analysis of relationships between variables",
            "data_quality": "Assessment of data quality and completeness",
            "business_insights": "Potential business or research implications",
            "recommendations": ["recommendation1", "recommendation2", ...],
            "anomalies": "Any unusual patterns or outliers detected"
        }}
        """
        
        # Check if OpenAI client is available
        if not openai_client:
            raise Exception("OpenAI API key not configured")
            
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data scientist expert who provides detailed, actionable insights from datasets. Analyze data patterns, correlations, and provide meaningful business insights."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        ai_insights = json.loads(response.choices[0].message.content)
        
        # Combine AI insights with technical summary
        result = {
            'ai_insights': ai_insights,
            'technical_summary': data_summary,
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Error generating insights: {str(e)}")
        # Return a fallback response if AI analysis fails
        df = pd.read_csv(csv_path)
        fallback_insights = {
            'ai_insights': {
                'summary': f"Dataset contains {len(df)} rows and {len(df.columns)} columns. AI analysis unavailable.",
                'key_findings': [
                    f"Dataset has {len(df.select_dtypes(include=[np.number]).columns)} numeric columns",
                    f"Dataset has {len(df.select_dtypes(include=['object']).columns)} categorical columns",
                    f"Total data points: {len(df) * len(df.columns)}"
                ],
                'correlations_analysis': "Correlation analysis requires AI service to be available.",
                'data_quality': f"Missing values detected in {df.isnull().sum().sum()} cells total.",
                'business_insights': "Detailed business insights require AI analysis to be functional.",
                'recommendations': [
                    "Verify data quality and completeness",
                    "Explore relationships between numeric variables",
                    "Consider visualization of key metrics"
                ],
                'anomalies': "Anomaly detection requires AI analysis to be available."
            },
            'technical_summary': {
                'shape': {'rows': len(df), 'columns': len(df.columns)},
                'columns': list(df.columns),
                'data_types': df.dtypes.astype(str).to_dict(),
                'error': str(e)
            },
            'generated_at': pd.Timestamp.now().isoformat()
        }
        return fallback_insights
