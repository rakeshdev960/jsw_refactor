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
# Import the bag detector
from bag_detector import initialize_detector, get_detector
# Import the line counter
from line_counter import initialize_counter, get_counter, CountingLine

CHANNEL_PREFIX = os.environ.get("RTSP_PREFIX", "rtsp://admin:Fidelis12@103.21.79.245:554/Streaming/Channels")
MEDIA_MTX_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mediartx/mediamtx.yaml'))

app = Flask(__name__)

# === Database Configuration ===
# WARNING: Hardcoding credentials is not secure for production!
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:admin@localhost/jsw_inv_mng' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy properly
db = SQLAlchemy()
db.init_app(app)
# ===========================

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}}) # Allow all origins for development

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
    
    def __repr__(self):
        return f'<BagMovement {self.movement_type} {self.quantity} bags at {self.timestamp}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster.name if self.cluster else None,
            'timestamp': self.timestamp.isoformat(),
            'movement_type': self.movement_type,
            'quantity': self.quantity
        }
# =====================

# === MediaMTX Helper Functions ===
def read_mediamtx_config():
    try:
        with open(MEDIA_MTX_CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {'paths': {}} # Return default structure if file not found
    except yaml.YAMLError:
        return {'paths': {}} # Handle potential YAML errors

def write_mediamtx_config(config):
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(MEDIA_MTX_CONFIG_PATH), exist_ok=True)
    with open(MEDIA_MTX_CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def restart_mediamtx():
    try:
        # Find mediamtx.exe path relative to this script
        mediamtx_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mediartx'))
        mediamtx_exe = os.path.join(mediamtx_dir, 'mediamtx.exe')
        mediamtx_yaml = MEDIA_MTX_CONFIG_PATH

        if not os.path.exists(mediamtx_exe):
            print(f"Error: mediamtx.exe not found at {mediamtx_exe}")
            return False
        if not os.path.exists(mediamtx_yaml):
             print(f"Error: mediamtx.yaml not found at {mediamtx_yaml}")
             # Attempt to create a default one?
             write_mediamtx_config({'paths': {}}) # Create a basic empty config
             print("Created a default empty mediamtx.yaml")
             # return False # Decide if we should proceed without a config

        # Kill any running mediamtx
        print("Attempting to kill existing mediamtx process...")
        # Use taskkill on Windows. For Linux/macOS use pkill or killall
        subprocess.call('taskkill /IM mediamtx.exe /F', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1) # Give it a moment to die

        # Start mediamtx in background
        print(f"Starting mediamtx: {mediamtx_exe} {mediamtx_yaml}")
        # Use Popen for non-blocking execution
        subprocess.Popen([mediamtx_exe, mediamtx_yaml], cwd=mediamtx_dir)
        print("MediaMTX restart command issued.")
        return True
    except Exception as e:
        print(f"Error restarting MediaMTX: {e}")
        return False
# ===============================

# === Cluster API Endpoints ===
@app.route('/clusters', methods=['GET'])
def get_clusters():
    try:
        clusters = Cluster.query.all()
        return jsonify([cluster.to_dict() for cluster in clusters])
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
        
@app.route('/clusters/<int:cluster_id>/movements', methods=['GET'])
def get_cluster_movements(cluster_id):
    try:
        # Get optional date filter (format: YYYY-MM-DD)
        date_str = request.args.get('date')
        if date_str:
            try:
                # Parse date and create date range for the entire day
                filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                start_time = datetime.combine(filter_date, datetime.min.time())
                end_time = datetime.combine(filter_date, datetime.max.time())
                
                movements = BagMovement.query.filter(
                    BagMovement.cluster_id == cluster_id,
                    BagMovement.timestamp >= start_time,
                    BagMovement.timestamp <= end_time
                ).order_by(BagMovement.timestamp.desc()).all()
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        else:
            # Get the latest 50 movements if no date specified
            movements = BagMovement.query.filter_by(cluster_id=cluster_id).order_by(BagMovement.timestamp.desc()).limit(50).all()
        
        # Calculate daily summary for IN and OUT
        in_total = sum(m.quantity for m in movements if m.movement_type == 'IN')
        out_total = sum(m.quantity for m in movements if m.movement_type == 'OUT')
        
        return jsonify({
            'movements': [m.to_dict() for m in movements],
            'summary': {
                'in_total': in_total,
                'out_total': out_total,
                'net_change': in_total - out_total
            }
        })
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
        
@app.route('/clusters/daily-summary', methods=['GET'])
def get_daily_summary():
    try:
        # Get optional date filter (default to today)
        date_str = request.args.get('date')
        try:
            if date_str:
                filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                filter_date = datetime.now().date()
                
            start_time = datetime.combine(filter_date, datetime.min.time())
            end_time = datetime.combine(filter_date, datetime.max.time())
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        # Get all movements for the day
        movements = BagMovement.query.filter(
            BagMovement.timestamp >= start_time,
            BagMovement.timestamp <= end_time
        ).all()
        
        # Group by cluster and calculate totals
        cluster_summaries = {}
        
        for movement in movements:
            cluster_id = movement.cluster_id
            
            if cluster_id not in cluster_summaries:
                cluster = Cluster.query.get(cluster_id)
                if not cluster:
                    continue
                    
                cluster_summaries[cluster_id] = {
                    'cluster_id': cluster_id,
                    'cluster_name': cluster.name,
                    'in_total': 0,
                    'out_total': 0
                }
            
            if movement.movement_type == 'IN':
                cluster_summaries[cluster_id]['in_total'] += movement.quantity
            else:  # OUT
                cluster_summaries[cluster_id]['out_total'] += movement.quantity
        
        # Calculate net change for each cluster
        for summary in cluster_summaries.values():
            summary['net_change'] = summary['in_total'] - summary['out_total']
        
        return jsonify({
            'date': filter_date.strftime('%Y-%m-%d'),
            'summaries': list(cluster_summaries.values()),
            'overall': {
                'in_total': sum(s['in_total'] for s in cluster_summaries.values()),
                'out_total': sum(s['out_total'] for s in cluster_summaries.values()),
                'net_change': sum(s['net_change'] for s in cluster_summaries.values())
            }
        })
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/clusters', methods=['POST'])
def create_cluster():
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({'error': 'Cluster name is required'}), 400

    name = data['name']
    initial_count = data.get('bag_count', 0)
    capacity = data.get('capacity', 1000)  # Default capacity of 1000 if not provided

    if Cluster.query.filter_by(name=name).first():
        return jsonify({'error': f'Cluster with name "{name}" already exists'}), 409 # Conflict

    try:
        new_cluster = Cluster(name=name, bag_count=initial_count, capacity=capacity)
        db.session.add(new_cluster)
        db.session.commit()
        return jsonify(new_cluster.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/clusters/<int:cluster_id>', methods=['PUT'])
def update_cluster_count(cluster_id):
    cluster = Cluster.query.get(cluster_id)
    if not cluster:
        return jsonify({'error': 'Cluster not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Update bag_count if provided
        if 'bag_count' in data:
            if not isinstance(data['bag_count'], int):
                return jsonify({'error': 'bag_count must be an integer'}), 400
            cluster.bag_count = data['bag_count']
            
        # Update capacity if provided
        if 'capacity' in data:
            if not isinstance(data['capacity'], int) or data['capacity'] <= 0:
                return jsonify({'error': 'capacity must be a positive integer'}), 400
            cluster.capacity = data['capacity']
            
        db.session.commit()
        return jsonify(cluster.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/clusters/<int:cluster_id>/movement', methods=['POST'])
def record_bag_movement(cluster_id):
    """Record bag movement (in or out) for a specific cluster"""
    cluster = Cluster.query.get(cluster_id)
    if not cluster:
        return jsonify({'error': 'Cluster not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    movement_type = data.get('movement_type', '').upper()
    quantity = data.get('quantity')
    
    # Validate movement type
    if movement_type not in ['IN', 'OUT']:
        return jsonify({'error': 'movement_type must be either "IN" or "OUT"'}), 400
    
    # Validate quantity
    if not isinstance(quantity, int) or quantity <= 0:
        return jsonify({'error': 'quantity must be a positive integer'}), 400
    
    # Prevent removing more bags than exist in the cluster
    if movement_type == 'OUT' and quantity > cluster.bag_count:
        return jsonify({'error': f'Cannot remove {quantity} bags. Only {cluster.bag_count} bags available in the cluster.'}), 400
    
    try:
        # Create bag movement record
        movement = BagMovement(
            cluster_id=cluster_id,
            movement_type=movement_type,
            quantity=quantity,
            timestamp=datetime.now()
        )
        
        # Update the cluster's bag count
        if movement_type == 'IN':
            cluster.bag_count += quantity
        else:  # OUT
            cluster.bag_count -= quantity
            
        db.session.add(movement)
        db.session.commit()
        
        return jsonify({
            'movement': movement.to_dict(),
            'cluster': cluster.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/clusters/<int:cluster_id>', methods=['DELETE']) # Added Delete
def delete_cluster(cluster_id):
    cluster = Cluster.query.get(cluster_id)
    if not cluster:
        return jsonify({'error': 'Cluster not found'}), 404
    try:
        db.session.delete(cluster)
        db.session.commit()
        return jsonify({'status': 'Cluster deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/clusters/<int:cluster_id>/reset', methods=['POST'])
def reset_cluster(cluster_id):
    """Reset bag count and movement history for a cluster"""
    cluster = Cluster.query.get(cluster_id)
    if not cluster:
        return jsonify({'error': 'Cluster not found'}), 404
    try:
        # Delete all movement history for this cluster
        BagMovement.query.filter_by(cluster_id=cluster_id).delete()
        
        # Reset the bag count to 0
        cluster.bag_count = 0
        
        # Commit all changes
        db.session.commit()
        
        return jsonify({
            'status': 'success', 
            'message': f'Reset bag count and movement history for {cluster.name}',
            'cluster': cluster.to_dict()
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {str(e)}'}), 500
# ===========================

# === Camera API Endpoints ===
@app.route('/cameras', methods=['GET'])
def get_cameras():
    config = read_mediamtx_config()
    paths = config.get('paths', {})
    cameras = []
    for name, details in paths.items():
        # Try to extract URL, handle potential missing 'source'
        url = details.get('source', 'URL not specified')
        cameras.append({'name': name, 'url': url})
    return jsonify(cameras)

# Endpoint to add camera by CHANNEL (original functionality)
@app.route('/add_camera_channel', methods=['POST'])
def add_camera_channel():
    data = request.get_json()
    cam_name = data.get('cam_name')
    channel = data.get('channel')
    if not cam_name or not channel:
        return jsonify({'error': 'cam_name and channel are required'}), 400

    rtsp_url = f"{CHANNEL_PREFIX}/{channel}"

    config = read_mediamtx_config()
    if 'paths' not in config:
        config['paths'] = {}

    # Avoid overwriting existing path
    if cam_name in config['paths']:
         return jsonify({'error': f'Camera name "{cam_name}" already exists'}), 409

    config['paths'][cam_name] = {
        'source': rtsp_url,
        'sourceOnDemand': True # Use True (boolean) instead of 'yes' for YAML
    }

    write_mediamtx_config(config)

    if restart_mediamtx():
        return jsonify({'status': f'Camera {cam_name} added via channel {channel} and MediaMTX restarting'}) , 201
    else:
        return jsonify({'error': 'Camera added to config, but failed to restart MediaMTX'}), 500

# New endpoint to add camera by FULL URL
@app.route('/add_camera_url', methods=['POST'])
def add_camera_url():
    data = request.get_json()
    cam_name = data.get('name')
    rtsp_url = data.get('url')
    if not cam_name or not rtsp_url:
        return jsonify({'error': 'Camera name and url are required'}), 400

    # Basic validation for RTSP URL format (can be improved)
    if not rtsp_url.lower().startswith('rtsp://'):
         return jsonify({'error': 'Invalid RTSP URL format'}), 400

    config = read_mediamtx_config()
    if 'paths' not in config:
        config['paths'] = {}

    # Avoid overwriting existing path
    if cam_name in config['paths']:
         return jsonify({'error': f'Camera name "{cam_name}" already exists'}), 409

    config['paths'][cam_name] = {
        'source': rtsp_url,
        'sourceOnDemand': True
    }

    write_mediamtx_config(config)

    if restart_mediamtx():
        return jsonify({'status': f'Camera {cam_name} added via URL and MediaMTX restarting'}), 201
    else:
        return jsonify({'error': 'Camera added to config, but failed to restart MediaMTX'}), 500

@app.route('/delete_camera/<string:cam_name>', methods=['DELETE'])
def delete_camera(cam_name):
    config = read_mediamtx_config()
    if 'paths' not in config or cam_name not in config['paths']:
        return jsonify({'error': f'Camera "{cam_name}" not found'}), 404

    del config['paths'][cam_name]
    write_mediamtx_config(config)

    if restart_mediamtx():
        return jsonify({'status': f'Camera {cam_name} deleted and MediaMTX restarting'}), 200
    else:
        return jsonify({'error': 'Camera deleted from config, but failed to restart MediaMTX'}), 500

# ==========================


# === Line Counting API Endpoints ===
@app.route('/cameras/<string:camera_name>/counting-line', methods=['GET'])
def get_counting_line(camera_name):
    """Get the counting line for a specific camera"""
    counter = get_counter()
    line = counter.get_line(camera_name)
    if line:
        return jsonify({
            'camera_name': camera_name,
            'line': line
        }), 200
    return jsonify({'error': f'No counting line defined for camera {camera_name}'}), 404

@app.route('/cameras/<string:camera_name>/counting-line', methods=['POST'])
def set_counting_line(camera_name):
    """Set a counting line for a specific camera"""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    start_point = data.get('start_point')
    end_point = data.get('end_point')
    in_direction = data.get('in_direction', 'right')
    
    if not start_point or not end_point:
        return jsonify({'error': 'start_point and end_point are required'}), 400
    
    # Check if camera exists
    config = read_mediamtx_config()
    if camera_name not in config.get('paths', {}):
        return jsonify({'error': f'Camera {camera_name} not found in MediaMTX config'}), 404
    
    # Set counting line
    counter = get_counter()
    line = counter.set_counting_line(camera_name, start_point, end_point, in_direction)
    
    return jsonify({
        'camera_name': camera_name,
        'line': line
    }), 200

@app.route('/cameras/<string:camera_name>/counting-line', methods=['DELETE'])
def delete_counting_line(camera_name):
    """Delete the counting line for a specific camera"""
    counter = get_counter()
    if counter.get_line(camera_name):
        # Delete the line from database
        try:
            with app.app_context():
                line = CountingLine.query.filter_by(camera_name=camera_name).first()
                if line:
                    db.session.delete(line)
                    db.session.commit()
        except Exception as e:
            return jsonify({'error': f'Database error: {str(e)}'}), 500
        
        # Reset counts for this camera
        counter.reset_counts(camera_name)
        # Remove the line from memory
        if camera_name in counter.counting_lines:
            counter.counting_lines.pop(camera_name)
        
        return jsonify({'status': f'Counting line for camera {camera_name} deleted'}), 200
    
    return jsonify({'error': f'No counting line defined for camera {camera_name}'}), 404

@app.route('/cameras/<string:camera_name>/counts', methods=['GET'])
def get_camera_counts(camera_name):
    """Get the current counts for a specific camera"""
    counter = get_counter()
    counts = counter.get_counts(camera_name)
    
    return jsonify({
        'camera_name': camera_name,
        'counts': counts
    }), 200

@app.route('/cameras/<string:camera_name>/counts/reset', methods=['POST'])
def reset_camera_counts(camera_name):
    """Reset the counts for a specific camera"""
    counter = get_counter()
    counter.reset_counts(camera_name)
    
    return jsonify({
        'camera_name': camera_name,
        'status': 'Counts reset successfully'
    }), 200

@app.route('/cameras/counts', methods=['GET'])
def get_all_counts():
    """Get counts for all cameras"""
    counter = get_counter()
    counts = counter.get_counts()
    
    return jsonify(counts), 200

# === Bag Detection API Endpoints ===
@app.route('/cameras/<string:camera_name>/detection/start', methods=['POST'])
def start_bag_detection(camera_name):
    """Start bag detection for a specific camera feed"""
    detector = get_detector()
    if camera_name not in detector.camera_feeds:
        # Try to add the camera first
        config = read_mediamtx_config()
        if camera_name not in config.get('paths', {}):
            return jsonify({'error': f'Camera {camera_name} not found in MediaMTX config'}), 404
        
        # Get RTSP URL from MediaMTX config
        rtsp_url = f"rtsp://localhost:8554/{camera_name}"
        success, message = detector.add_camera_feed(camera_name, rtsp_url)
        if not success:
            return jsonify({'error': message}), 400
    
    success, message = detector.start_detection(camera_name)
    if success:
        return jsonify({'status': message}), 200
    return jsonify({'error': message}), 400

@app.route('/cameras/<string:camera_name>/detection/stop', methods=['POST'])
def stop_bag_detection(camera_name):
    """Stop bag detection for a specific camera feed"""
    detector = get_detector()
    success, message = detector.stop_detection(camera_name)
    if success:
        return jsonify({'status': message}), 200
    return jsonify({'error': message}), 400

@app.route('/cameras/<string:camera_name>/detection/results', methods=['GET'])
def get_bag_detection_results(camera_name):
    """Get bag detection results for a specific camera feed"""
    detector = get_detector()
    results = detector.get_detection_results(camera_name)
    if results:
        return jsonify(results), 200
    return jsonify({'error': f'Camera {camera_name} not found'}), 404

@app.route('/cameras/detection/results', methods=['GET'])
def get_all_detection_results():
    """Get bag detection results for all camera feeds"""
    detector = get_detector()
    results = detector.get_detection_results()
    return jsonify(results), 200

@app.route('/cameras/<string:camera_name>/detection/frame', methods=['GET'])
def get_detection_frame(camera_name):
    """Get the latest annotated frame with bag detections"""
    detector = get_detector()
    
    # Get quality parameter from query string, default to 'low' for faster loading
    quality = request.args.get('quality', 'low')
    
    # Check if camera exists in MediaMTX but not in detector
    if camera_name not in detector.camera_feeds:
        config = read_mediamtx_config()
        if camera_name in config.get('paths', {}):
            # Add camera to detector without enabling detection
            rtsp_url = f"rtsp://localhost:8554/{camera_name}"
            detector.add_camera_feed(camera_name, rtsp_url)
            # Start a thread to just capture frames (detection will be disabled by default)
            thread = threading.Thread(
                target=detector._process_camera_feed,
                args=(camera_name,),
                daemon=True
            )
            detector.processing_threads[camera_name] = thread
            thread.start()
            # Give it a moment to start capturing frames
            return jsonify({'status': 'Camera feed initializing, please try again in a few seconds'}), 202
    
    # Try to get the latest frame with requested quality
    frame_bytes = detector.get_latest_annotated_frame(camera_name, quality=quality)
    
    if frame_bytes:
        return Response(frame_bytes, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'No frame available for camera ' + camera_name}), 404

# ==================================


# === Initialization ===
with app.app_context():
    print("Creating database tables if they don't exist...")
    try:
        db.create_all()
        print("Database tables checked/created.")
        
        # Initialize the bag detector
        print("Initializing bag detector...")
        initialize_detector(confidence_threshold=0.5)
        print("Bag detector initialized.")
        
        # Initialize the line counter
        print("Initializing line counter...")
        initialize_counter()
        print("Line counter initialized.")
    except Exception as e:
        print(f"Error during initialization: {e}")
        print("Please ensure MySQL server is running and database 'jsw_inv_mng' exists.")
        # Optionally exit or handle differently
# ======================

if __name__ == '__main__':
    print("Starting Flask server...")
    # Use host='0.0.0.0' to be accessible externally, debug=False for production
    app.run(host='0.0.0.0', port=5000, debug=True)
