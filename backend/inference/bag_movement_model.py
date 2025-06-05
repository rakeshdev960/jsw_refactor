from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class BagMovement(db.Model):
    """
    Model to track bag movements (in/out) for each cluster
    """
    id = db.Column(db.Integer, primary_key=True)
    cluster_id = db.Column(db.Integer, db.ForeignKey('cluster.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now, nullable=False)
    movement_type = db.Column(db.String(10), nullable=False)  # 'IN' or 'OUT'
    quantity = db.Column(db.Integer, default=0, nullable=False)
    
    def __repr__(self):
        return f'<BagMovement {self.movement_type} {self.quantity} bags at {self.timestamp}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'cluster_id': self.cluster_id,
            'timestamp': self.timestamp.isoformat(),
            'movement_type': self.movement_type,
            'quantity': self.quantity
        }
