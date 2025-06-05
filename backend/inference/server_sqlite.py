from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import subprocess
import threading
import time
import os
import yaml
import cv2
import numpy as np

app = Flask(__name__)

# Use SQLite for simplicity
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jsw_inventory.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# === Database Models ===
class Cluster(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    bag_count = db.Column(db.Integer, default=0)
    capacity = db.Column(db.Integer, default=1000)  # Default capacity of 1000 bags
    bag_movements = db.relationship('BagMovement', backref='cluster', lazy=True)

    def __repr__(self):
        return f'<Cluster {self.name}>'

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'bag_count': self.bag_count,
            'capacity': self.capacity,
            'utilization': round((self.bag_count / self.capacity * 100), 1) if self.capacity > 0 else 0
        }

class BagMovement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    cluster_id = db.Column(db.Integer, db.ForeignKey('cluster.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now, nullable=False)
    movement_type = db.Column(db.String(10), nullable=False)  # 'IN' or 'OUT'
    quantity = db.Column(db.Integer, default=0, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'cluster_id': self.cluster_id,
            'timestamp': self.timestamp.isoformat(),
            'movement_type': self.movement_type,
            'quantity': self.quantity
        }

# === Routes ===

# Get all clusters
@app.route('/clusters', methods=['GET'])
def get_clusters():
    clusters = Cluster.query.all()
    return jsonify([cluster.to_dict() for cluster in clusters])

# Create a new cluster
@app.route('/clusters', methods=['POST'])
def create_cluster():
    data = request.json
    if not data or 'name' not in data:
        return jsonify({'error': 'Name is required'}), 400
    
    name = data.get('name')
    bag_count = data.get('bag_count', 0)
    capacity = data.get('capacity', 1000)
    
    # Check if cluster name already exists
    existing_cluster = Cluster.query.filter_by(name=name).first()
    if existing_cluster:
        return jsonify({'error': f'Cluster with name {name} already exists'}), 400
    
    new_cluster = Cluster(name=name, bag_count=bag_count, capacity=capacity)
    db.session.add(new_cluster)
    db.session.commit()
    
    return jsonify(new_cluster.to_dict()), 201

# Get a specific cluster
@app.route('/clusters/<int:cluster_id>', methods=['GET'])
def get_cluster(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    return jsonify(cluster.to_dict())

# Update an existing cluster
@app.route('/clusters/<int:cluster_id>', methods=['PUT'])
def update_cluster(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    data = request.json
    
    if 'name' in data:
        # Check if the new name already exists for a different cluster
        existing_cluster = Cluster.query.filter_by(name=data['name']).first()
        if existing_cluster and existing_cluster.id != cluster_id:
            return jsonify({'error': f'Cluster with name {data["name"]} already exists'}), 400
        cluster.name = data['name']
    
    if 'bag_count' in data:
        cluster.bag_count = data['bag_count']
    
    if 'capacity' in data:
        cluster.capacity = data['capacity']
    
    db.session.commit()
    return jsonify(cluster.to_dict())

# Delete a cluster
@app.route('/clusters/<int:cluster_id>', methods=['DELETE'])
def delete_cluster(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    
    # Delete all associated bag movements first
    BagMovement.query.filter_by(cluster_id=cluster_id).delete()
    
    db.session.delete(cluster)
    db.session.commit()
    
    return jsonify({'message': f'Cluster {cluster.name} deleted successfully'})

# Record a bag movement for a cluster
@app.route('/clusters/<int:cluster_id>/movement', methods=['POST'])
def add_movement(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    data = request.json
    
    if not data or 'movement_type' not in data or 'quantity' not in data:
        return jsonify({'error': 'Movement type and quantity are required'}), 400
    
    movement_type = data['movement_type'].upper()
    quantity = int(data['quantity'])
    
    if movement_type not in ['IN', 'OUT']:
        return jsonify({'error': 'Movement type must be IN or OUT'}), 400
    
    if quantity <= 0:
        return jsonify({'error': 'Quantity must be positive'}), 400
    
    # Create movement record
    movement = BagMovement(
        cluster_id=cluster_id,
        movement_type=movement_type,
        quantity=quantity
    )
    
    # Update cluster bag count
    if movement_type == 'IN':
        cluster.bag_count += quantity
    else:  # 'OUT'
        if cluster.bag_count < quantity:
            return jsonify({'error': f'Not enough bags in cluster. Current count: {cluster.bag_count}'}), 400
        cluster.bag_count -= quantity
    
    db.session.add(movement)
    db.session.commit()
    
    return jsonify({
        'message': f'{quantity} bags {movement_type} recorded successfully',
        'movement': movement.to_dict(),
        'cluster': cluster.to_dict()
    }), 201

# Get movement history for a cluster
@app.route('/clusters/<int:cluster_id>/movements', methods=['GET'])
def get_movements(cluster_id):
    Cluster.query.get_or_404(cluster_id)  # Check if cluster exists
    
    movements = BagMovement.query.filter_by(cluster_id=cluster_id).order_by(BagMovement.timestamp.desc()).all()
    return jsonify([movement.to_dict() for movement in movements])

# Get daily movement summary for all clusters
@app.route('/clusters/daily-summary', methods=['GET'])
def get_daily_summary():
    date_str = request.args.get('date', None)
    
    if date_str:
        try:
            # Parse the date string to datetime
            from datetime import datetime, time
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Create datetime objects for start and end of the day
            start_of_day = datetime.combine(date_obj, time.min)
            end_of_day = datetime.combine(date_obj, time.max)
            
            # Query movements for this day
            movements = BagMovement.query.filter(
                BagMovement.timestamp >= start_of_day,
                BagMovement.timestamp <= end_of_day
            ).all()
            
            # Group by cluster and movement type
            summary = {}
            for movement in movements:
                cluster = Cluster.query.get(movement.cluster_id)
                if not cluster:
                    continue
                    
                if cluster.id not in summary:
                    summary[cluster.id] = {
                        'cluster_id': cluster.id,
                        'cluster_name': cluster.name,
                        'in_count': 0,
                        'out_count': 0,
                        'net_change': 0,
                        'current_count': cluster.bag_count
                    }
                
                if movement.movement_type == 'IN':
                    summary[cluster.id]['in_count'] += movement.quantity
                elif movement.movement_type == 'OUT':
                    summary[cluster.id]['out_count'] += movement.quantity
                
                summary[cluster.id]['net_change'] = summary[cluster.id]['in_count'] - summary[cluster.id]['out_count']
            
            return jsonify(list(summary.values()))
        except Exception as e:
            return jsonify({'error': f'Error processing date: {str(e)}'}), 400
    else:
        # Return all clusters with zeros if no date provided
        clusters = Cluster.query.all()
        summary = [{
            'cluster_id': cluster.id,
            'cluster_name': cluster.name,
            'in_count': 0,
            'out_count': 0,
            'net_change': 0,
            'current_count': cluster.bag_count
        } for cluster in clusters]
        
        return jsonify(summary)

# Reset a cluster (set bag count to 0 and delete movement history)
@app.route('/clusters/<int:cluster_id>/reset', methods=['POST'])
def reset_cluster(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    
    # Delete all bag movements for this cluster
    BagMovement.query.filter_by(cluster_id=cluster_id).delete()
    
    # Reset bag count to 0
    cluster.bag_count = 0
    
    db.session.commit()
    
    return jsonify({
        'message': f'Cluster {cluster.name} has been reset',
        'cluster': cluster.to_dict()
    })

# Root route for API info
@app.route('/')
def index():
    return jsonify({
        'name': 'JSW Cement Bag Detection API',
        'version': '1.0',
        'endpoints': {
            'GET /clusters': 'List all clusters',
            'POST /clusters': 'Create a new cluster',
            'GET /clusters/<id>': 'Get a specific cluster',
            'PUT /clusters/<id>': 'Update a cluster',
            'DELETE /clusters/<id>': 'Delete a cluster',
            'POST /clusters/<id>/movement': 'Record bag movement for a cluster',
            'GET /clusters/<id>/movements': 'Get movement history for a cluster',
            'POST /clusters/<id>/reset': 'Reset cluster bag count and movement history'
        }
    })

# Create database tables on startup
with app.app_context():
    print("Creating database tables if they don't exist...")
    db.create_all()
    print("Database tables checked/created.")

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
