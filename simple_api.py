from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jsw_inventory.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Enable CORS
CORS(app)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Models
class Cluster(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    bag_count = db.Column(db.Integer, default=0)
    capacity = db.Column(db.Integer, default=1000)
    
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
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    movement_type = db.Column(db.String(10), nullable=False)  # 'IN' or 'OUT'
    quantity = db.Column(db.Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'cluster_id': self.cluster_id,
            'timestamp': self.timestamp.isoformat(),
            'movement_type': self.movement_type,
            'quantity': self.quantity
        }

# Create tables
with app.app_context():
    db.create_all()
    print("Database tables created/verified.")

# Routes
@app.route('/clusters', methods=['GET'])
def get_clusters():
    clusters = Cluster.query.all()
    return jsonify([cluster.to_dict() for cluster in clusters])

@app.route('/clusters', methods=['POST'])
def create_cluster():
    data = request.json
    if not data or 'name' not in data:
        return jsonify({'error': 'Name is required'}), 400
    
    name = data.get('name')
    bag_count = data.get('bag_count', 0)
    capacity = data.get('capacity', 1000)
    
    existing = Cluster.query.filter_by(name=name).first()
    if existing:
        return jsonify({'error': 'Cluster with this name already exists'}), 400
    
    cluster = Cluster(name=name, bag_count=bag_count, capacity=capacity)
    db.session.add(cluster)
    db.session.commit()
    
    return jsonify(cluster.to_dict()), 201

@app.route('/clusters/<int:cluster_id>', methods=['GET'])
def get_cluster(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    return jsonify(cluster.to_dict())

@app.route('/clusters/<int:cluster_id>', methods=['PUT'])
def update_cluster(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    data = request.json
    
    if 'bag_count' in data:
        cluster.bag_count = data['bag_count']
    if 'capacity' in data:
        cluster.capacity = data['capacity']
    if 'name' in data:
        existing = Cluster.query.filter_by(name=data['name']).first()
        if existing and existing.id != cluster_id:
            return jsonify({'error': 'Cluster with this name already exists'}), 400
        cluster.name = data['name']
    
    db.session.commit()
    return jsonify(cluster.to_dict())

@app.route('/clusters/<int:cluster_id>', methods=['DELETE'])
def delete_cluster(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    db.session.delete(cluster)
    db.session.commit()
    return jsonify({'message': 'Cluster deleted successfully'})

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
    
    movement = BagMovement(
        cluster_id=cluster.id,
        movement_type=movement_type,
        quantity=quantity
    )
    
    db.session.add(movement)
    
    # Update cluster bag count
    if movement_type == 'IN':
        cluster.bag_count += quantity
    else:  # OUT
        cluster.bag_count = max(0, cluster.bag_count - quantity)
    
    db.session.commit()
    
    return jsonify({
        'movement': movement.to_dict(),
        'cluster': cluster.to_dict()
    }), 201

@app.route('/clusters/<int:cluster_id>/movements', methods=['GET'])
def get_movements(cluster_id):
    cluster = Cluster.query.get_or_404(cluster_id)
    movements = BagMovement.query.filter_by(cluster_id=cluster.id).order_by(BagMovement.timestamp.desc()).all()
    return jsonify([movement.to_dict() for movement in movements])

@app.route('/clusters/<int:cluster_id>/reset', methods=['POST'])
def reset_cluster(cluster_id):
    """Reset a cluster's bag count to 0 and delete all movement history"""
    cluster = Cluster.query.get_or_404(cluster_id)
    
    # Delete all movements for this cluster
    BagMovement.query.filter_by(cluster_id=cluster.id).delete()
    
    # Reset bag count
    cluster.bag_count = 0
    
    db.session.commit()
    
    return jsonify({
        'message': f'Cluster {cluster.name} has been reset',
        'cluster': cluster.to_dict()
    })

# Test route
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'JSW Inventory API is running',
        'endpoints': {
            'GET /clusters': 'List all clusters',
            'POST /clusters': 'Create a new cluster',
            'GET /clusters/<id>': 'Get a specific cluster',
            'PUT /clusters/<id>': 'Update a cluster',
            'DELETE /clusters/<id>': 'Delete a cluster',
            'POST /clusters/<id>/movement': 'Record movement for a cluster',
            'GET /clusters/<id>/movements': 'Get movement history for a cluster',
            'POST /clusters/<id>/reset': 'Reset cluster count and history'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
