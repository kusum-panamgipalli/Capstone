"""
Web Integration Framework for ISL Interpreter
This script provides a WebSocket server for browser integration
"""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from tensorflow import keras
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISLWebServer:
    def __init__(self, model_path, labels_path, host='localhost', port=8765):
        """Initialize the WebSocket server for ISL interpretation"""
        self.host = host
        self.port = port
        
        # Load model and labels
        logger.info("Loading model...")
        self.model = keras.models.load_model(model_path)
        
        with open(labels_path, 'r') as f:
            self.class_names = json.load(f)
        
        logger.info(f"✓ Model loaded with {len(self.class_names)} classes")
        
        self.img_size = 128
        self.clients = set()
    
    def preprocess_frame(self, frame_data):
        """Preprocess base64 encoded frame"""
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize and normalize
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, frame_data):
        """Make prediction on frame"""
        try:
            processed = self.preprocess_frame(frame_data)
            predictions = self.model.predict(processed, verbose=0)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            results = {
                'success': True,
                'predictions': [
                    {
                        'class': self.class_names[idx],
                        'confidence': float(predictions[idx])
                    }
                    for idx in top_indices
                ]
            }
            
            return results
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        client_id = id(websocket)
        self.clients.add(websocket)
        logger.info(f"Client {client_id} connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'frame':
                    # Process frame and send prediction
                    result = self.predict(data['data'])
                    await websocket.send(json.dumps(result))
                
                elif data['type'] == 'ping':
                    # Health check
                    await websocket.send(json.dumps({'type': 'pong'}))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client {client_id} removed. Total clients: {len(self.clients)}")
    
    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting ISL WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"✓ Server running at ws://{self.host}:{self.port}")
            logger.info("Waiting for browser connections...")
            await asyncio.Future()  # Run forever

def main():
    """Main entry point"""
    MODEL_PATH = 'isl_model.h5'
    LABELS_PATH = 'isl_labels.json'
    
    server = ISLWebServer(MODEL_PATH, LABELS_PATH)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")

if __name__ == "__main__":
    main()
