from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class BoundaryBox(db.Model):
    """
    Model to store custom boundary boxes for cameras to define counting areas
    """
    id = db.Column(db.Integer, primary_key=True)
    camera_name = db.Column(db.String(80), unique=True, nullable=False)
    # Store boundary boxes as JSON: [{"name": "Zone 1", "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "color": "#ff0000"}]
    boundaries = db.Column(db.Text, nullable=False, default='[]')  
    created_at = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    
    def __repr__(self):
        return f'<BoundaryBox {self.camera_name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'camera_name': self.camera_name,
            'boundaries': json.loads(self.boundaries),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @staticmethod
    def get_boundaries(camera_name):
        """Get boundary boxes for a camera"""
        box = BoundaryBox.query.filter_by(camera_name=camera_name).first()
        if box:
            return json.loads(box.boundaries)
        return []
    
    @staticmethod
    def save_boundaries(camera_name, boundaries):
        """Save boundary boxes for a camera"""
        box = BoundaryBox.query.filter_by(camera_name=camera_name).first()
        if not box:
            box = BoundaryBox(camera_name=camera_name)
            
        box.boundaries = json.dumps(boundaries)
        db.session.add(box)
        db.session.commit()
        return box.to_dict()
