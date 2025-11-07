// Chrome Extension - Content Script for Google Meet Integration
// This is a template for future web extension development

class ISLOverlay {
    constructor() {
        this.websocket = null;
        this.videoElement = null;
        this.canvas = null;
        this.overlayDiv = null;
        this.isActive = false;
        this.currentTranslation = '';
        
        this.init();
    }
    
    init() {
        // Create overlay UI
        this.createOverlay();
        
        // Connect to WebSocket server
        this.connectToServer();
        
        // Find video element (Google Meet specific)
        this.findVideoElement();
        
        // Start capturing frames
        this.startCapture();
    }
    
    createOverlay() {
        // Create overlay div for displaying translations
        this.overlayDiv = document.createElement('div');
        this.overlayDiv.id = 'isl-overlay';
        this.overlayDiv.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: #00ff00;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 24px;
            font-family: monospace;
            z-index: 10000;
            min-width: 400px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        `;
        this.overlayDiv.textContent = 'ISL Interpreter: Ready';
        document.body.appendChild(this.overlayDiv);
        
        // Create canvas for frame capture
        this.canvas = document.createElement('canvas');
        this.canvas.width = 640;
        this.canvas.height = 480;
    }
    
    connectToServer() {
        try {
            this.websocket = new WebSocket('ws://localhost:8765');
            
            this.websocket.onopen = () => {
                console.log('Connected to ISL server');
                this.updateOverlay('ISL Interpreter: Connected', '#00ff00');
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handlePrediction(data);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateOverlay('ISL Interpreter: Connection Error', '#ff0000');
            };
            
            this.websocket.onclose = () => {
                console.log('Disconnected from ISL server');
                this.updateOverlay('ISL Interpreter: Disconnected', '#ff9900');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectToServer(), 5000);
            };
        } catch (error) {
            console.error('Failed to connect:', error);
        }
    }
    
    findVideoElement() {
        // Try to find video element (adjust selector for different platforms)
        // Google Meet: 'video[autoplay]'
        // Zoom: Different selector needed
        
        const checkForVideo = setInterval(() => {
            const videos = document.querySelectorAll('video');
            
            // Find the user's own video (usually smaller)
            for (let video of videos) {
                if (video.videoWidth > 0 && video.videoHeight > 0) {
                    this.videoElement = video;
                    console.log('Video element found:', video);
                    clearInterval(checkForVideo);
                    break;
                }
            }
        }, 1000);
    }
    
    captureFrame() {
        if (!this.videoElement || !this.canvas) return null;
        
        const ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.videoElement, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to base64
        return this.canvas.toDataURL('image/jpeg', 0.8);
    }
    
    startCapture() {
        // Capture and send frames at 5 FPS (adjust as needed)
        setInterval(() => {
            if (this.isActive && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                const frameData = this.captureFrame();
                
                if (frameData) {
                    this.websocket.send(JSON.stringify({
                        type: 'frame',
                        data: frameData
                    }));
                }
            }
        }, 200); // 5 FPS
    }
    
    handlePrediction(data) {
        if (data.success && data.predictions && data.predictions.length > 0) {
            const topPrediction = data.predictions[0];
            
            // Only update if confidence is high enough
            if (topPrediction.confidence > 0.7) {
                this.currentTranslation += topPrediction.class;
                this.updateOverlay(`Translation: ${this.currentTranslation}`);
            }
        }
    }
    
    updateOverlay(text, color = '#00ff00') {
        if (this.overlayDiv) {
            this.overlayDiv.textContent = text;
            this.overlayDiv.style.color = color;
        }
    }
    
    toggle() {
        this.isActive = !this.isActive;
        this.updateOverlay(
            this.isActive ? 'ISL Interpreter: Active' : 'ISL Interpreter: Paused',
            this.isActive ? '#00ff00' : '#ff9900'
        );
    }
    
    clearTranslation() {
        this.currentTranslation = '';
        this.updateOverlay('Translation: [Cleared]');
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.islOverlay = new ISLOverlay();
    });
} else {
    window.islOverlay = new ISLOverlay();
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Alt + I: Toggle ISL interpreter
    if (e.altKey && e.key === 'i') {
        window.islOverlay.toggle();
    }
    
    // Alt + C: Clear translation
    if (e.altKey && e.key === 'c') {
        window.islOverlay.clearTranslation();
    }
});
