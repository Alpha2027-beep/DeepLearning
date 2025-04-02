from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import base64
from deepfake_detector import DeepfakeEnsembleDetector

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model paths
MODEL_PATHS = {
    'deepfake_model1.pth': 'models/deepfake_model1.pth',
    'deepfake_model2.pth': 'models/deepfake_model2.pth',
    'deepfake_detector.pth': 'models/deepfake_detector.pth',
    'best_realfake_model.pth': 'models/best_realfake_model.pth',
    'best_ensemble_model.pth': 'models/best_ensemble_model.pth'
}

# Initialize detector
try:
    detector = DeepfakeEnsembleDetector(MODEL_PATHS)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    detector = None

def process_prediction(results):
    """Process and format prediction results"""
    return {
        'is_fake': bool(results['is_fake']),
        'confidence': float(results['ensemble_prediction']),
        'model_predictions': [bool(pred) for pred in results['model_predictions']],
        'model_confidences': [float(conf) for conf in results['model_confidences']],
        'fake_votes': int(results['fake_votes']),
        'total_models': int(results['total_models'])
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process image
    try:
        image = Image.open(filepath)
        results = detector.predict(image)
        return jsonify(process_prediction(results))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process video
    try:
        video = cv2.VideoCapture(filepath)
        results = []
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Detect every 10th frame to reduce processing time
            frame_results = detector.predict(frame)
            results.append(process_prediction(frame_results))
        
        video.release()
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/live_detection', methods=['POST'])
def live_detection():
    """Handle live camera stream detection"""
    try:
        # Get base64 encoded image from request
        image_data = request.json.get('image', '')
        
        # Decode base64 image
        image_data = base64.b64decode(image_data.split(',')[1])
        image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Detect
        results = detector.predict(image_np)
        return jsonify(process_prediction(results))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)