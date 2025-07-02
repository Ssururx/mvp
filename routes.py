import os
import logging
from flask import render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
from app import app, db
from models import UploadedFile
from data_cleaner import clean_data
from analysis.overview import get_overview
from analysis.graphs import generate_graph_data
from analysis.equations import find_equation
from analysis.insights import generate_insights

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Only CSV files are allowed', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(pd.Timestamp.now().timestamp()))
        unique_filename = f"{timestamp}_{filename}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Create database record
        uploaded_file = UploadedFile(
            filename=unique_filename,
            original_filename=filename,
            file_size=os.path.getsize(filepath),
            status='uploaded'
        )
        db.session.add(uploaded_file)
        db.session.commit()
        
        # Clean the data
        try:
            cleaned_filename = clean_data(filepath, uploaded_file.id)
            uploaded_file.cleaned_filename = cleaned_filename
            uploaded_file.status = 'cleaned'
            db.session.commit()
            
            flash('File uploaded and cleaned successfully!', 'success')
            return redirect(url_for('analysis', file_id=uploaded_file.id))
            
        except Exception as e:
            logging.error(f"Error cleaning data: {str(e)}")
            uploaded_file.status = 'error'
            uploaded_file.error_message = str(e)
            db.session.commit()
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logging.error(f"Error uploading file: {str(e)}")
        flash(f'Error uploading file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analysis/<int:file_id>')
def analysis(file_id):
    uploaded_file = UploadedFile.query.get_or_404(file_id)
    
    if uploaded_file.status != 'cleaned':
        flash('File is not ready for analysis', 'error')
        return redirect(url_for('index'))
    
    return render_template('analysis.html', file=uploaded_file)

# API Endpoints
@app.route('/api/overview/<int:file_id>')
def api_overview(file_id):
    try:
        uploaded_file = UploadedFile.query.get_or_404(file_id)
        cleaned_path = os.path.join(app.config['CLEANED_FOLDER'], uploaded_file.cleaned_filename)
        
        overview_data = get_overview(cleaned_path)
        return jsonify(overview_data)
    except Exception as e:
        logging.error(f"Error getting overview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/<int:file_id>')
def api_graph(file_id):
    try:
        uploaded_file = UploadedFile.query.get_or_404(file_id)
        cleaned_path = os.path.join(app.config['CLEANED_FOLDER'], uploaded_file.cleaned_filename)
        
        x_column = request.args.get('x')
        y_column = request.args.get('y')
        
        if not x_column or not y_column:
            return jsonify({'error': 'x and y parameters are required'}), 400
        
        graph_data = generate_graph_data(cleaned_path, x_column, y_column)
        return jsonify(graph_data)
    except Exception as e:
        logging.error(f"Error generating graph: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/equation/<int:file_id>')
def api_equation(file_id):
    try:
        uploaded_file = UploadedFile.query.get_or_404(file_id)
        cleaned_path = os.path.join(app.config['CLEANED_FOLDER'], uploaded_file.cleaned_filename)
        
        x_column = request.args.get('x')
        y_column = request.args.get('y')
        
        if not x_column or not y_column:
            return jsonify({'error': 'x and y parameters are required'}), 400
        
        equation_data = find_equation(cleaned_path, x_column, y_column)
        return jsonify(equation_data)
    except Exception as e:
        logging.error(f"Error finding equation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/<int:file_id>')
def api_insights(file_id):
    try:
        uploaded_file = UploadedFile.query.get_or_404(file_id)
        cleaned_path = os.path.join(app.config['CLEANED_FOLDER'], uploaded_file.cleaned_filename)
        
        insights_data = generate_insights(cleaned_path)
        return jsonify(insights_data)
    except Exception as e:
        logging.error(f"Error generating insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('upload.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('upload.html'), 500
