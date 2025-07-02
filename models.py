from app import db
from datetime import datetime

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    cleaned_filename = db.Column(db.String(255))
    file_size = db.Column(db.Integer)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='uploaded')  # uploaded, cleaning, cleaned, error
    error_message = db.Column(db.Text)
    
    def __repr__(self):
        return f'<UploadedFile {self.original_filename}>'
