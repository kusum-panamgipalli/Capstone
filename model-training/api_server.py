"""
Flask API Server for ISL Sign Language Recognition
Serves predictions from the trained model to the Chrome extension
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import json
import os
from tensorflow import keras

app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome extension

# Load the trained model and preprocessing objects
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
model = None
scaler = None
label_mapping = None

def load_model_and_scaler():
    """Load the trained model, scaler, and label mapping"""
    global model, scaler, label_mapping
    
    print("Loading model and preprocessing objects...")
    
    # Load the trained Keras model
    model_path = os.path.join(MODEL_DIR, 'isl_model.h5')
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Load the scaler
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded from {scaler_path}")
    
    # Load label mapping
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        label_mapping = metadata['idx_to_label']
    print(f"✓ Label mapping loaded: {len(label_mapping)} classes")
    print(f"  Classes: {', '.join(sorted(label_mapping.values()))}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': len(label_mapping) if label_mapping else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sign from hand landmarks
    
    Expected JSON format:
    {
        "landmarks": [[x1, y1, z1], [x2, y2, z2], ..., [x21, y21, z21]]
    }
    
    Returns:
    {
        "sign": "A",
        "confidence": 0.98,
        "all_predictions": {"A": 0.98, "B": 0.01, ...}
    }
    """
    try:
        # Get landmarks from request
        data = request.get_json()
        
        if not data or 'landmarks' not in data:
            return jsonify({
                'error': 'Missing landmarks in request',
                'message': 'Please provide landmarks array'
            }), 400
        
        landmarks = data['landmarks']
        
        # Validate landmarks format
        if not isinstance(landmarks, list) or len(landmarks) != 21:
            return jsonify({
                'error': 'Invalid landmarks format',
                'message': 'Expected 21 landmarks (x, y, z) coordinates'
            }), 400
        
        # Flatten landmarks to feature vector (63 features: 21 landmarks × 3 coordinates)
        features = []
        for landmark in landmarks:
            if not isinstance(landmark, list) or len(landmark) != 3:
                return jsonify({
                    'error': 'Invalid landmark format',
                    'message': 'Each landmark should have [x, y, z] coordinates'
                }), 400
            features.extend(landmark)
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale features using the same scaler from training
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predictions = model.predict(features_scaled, verbose=0)[0]
        
        # Get the predicted class
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        
        # Map index to label (label_mapping is already idx_to_label with string keys)
        predicted_sign = label_mapping[str(predicted_class_idx)]
        
        # Get all predictions (top 5)
        top_indices = np.argsort(predictions)[-5:][::-1]
        all_predictions = {
            label_mapping[str(idx)]: float(predictions[idx])
            for idx in top_indices
        }
        
        return jsonify({
            'sign': predicted_sign,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of all supported sign classes"""
    return jsonify({
        'classes': sorted(label_mapping.keys()) if label_mapping else [],
        'num_classes': len(label_mapping) if label_mapping else 0
    })

if __name__ == '__main__':
    # Load model at startup
    load_model_and_scaler()
    
    # Run the server
    print("\n" + "="*60)
    print("ISL Sign Language Recognition API Server")
    print("="*60)
    print(f"Server running at: http://localhost:5000")
    print(f"Health check: http://localhost:5000/health")
    print(f"Prediction endpoint: POST http://localhost:5000/predict")
    print(f"Classes endpoint: GET http://localhost:5000/classes")
    print("\nReady to receive predictions from Chrome extension!")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
