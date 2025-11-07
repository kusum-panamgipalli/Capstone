// MediaPipe Hands integration for ISL Interpreter

class MediaPipeHandsProcessor {
    constructor() {
        this.hands = null;
        this.camera = null;
        this.isInitialized = false;
        this.isLoading = false;
        this.lastResults = null;
        this.onResultsCallback = null;
        
        // Configuration
        this.config = {
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
            maxNumHands: 2
        };
    }

    async initialize() {
        if (this.isInitialized || this.isLoading) {
            return this.isInitialized;
        }

        this.isLoading = true;
        console.log('Initializing MediaPipe Hands...');

        try {
            // Load MediaPipe scripts
            await this.loadMediaPipeScripts();

            // Initialize Hands
            this.hands = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`;
                }
            });

            // Configure hands
            await this.hands.setOptions(this.config);

            // Set up results callback
            this.hands.onResults((results) => {
                this.lastResults = results;
                if (this.onResultsCallback) {
                    this.onResultsCallback(results);
                }
            });

            this.isInitialized = true;
            console.log('MediaPipe Hands initialized successfully');
            return true;

        } catch (error) {
            console.error('Failed to initialize MediaPipe Hands:', error);
            this.isInitialized = false;
            return false;
        } finally {
            this.isLoading = false;
        }
    }

    async loadMediaPipeScripts() {
        const scripts = [
            'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1640029074/camera_utils.js',
            'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1620248257/drawing_utils.js',
            'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.js'
        ];

        for (const scriptUrl of scripts) {
            await this.loadScript(scriptUrl);
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            // Check if script already loaded
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async processFrame(videoElement) {
        if (!this.isInitialized) {
            console.warn('MediaPipe Hands not initialized');
            return null;
        }

        if (!videoElement || videoElement.videoWidth === 0) {
            return null;
        }

        try {
            // Send frame to MediaPipe
            await this.hands.send({ image: videoElement });
            
            // Return last processed results
            return this.lastResults;

        } catch (error) {
            console.error('Error processing frame with MediaPipe:', error);
            return null;
        }
    }

    setOnResultsCallback(callback) {
        this.onResultsCallback = callback;
    }

    getHandLandmarks() {
        if (!this.lastResults || !this.lastResults.multiHandLandmarks) {
            return [];
        }
        return this.lastResults.multiHandLandmarks;
    }

    getHandedness() {
        if (!this.lastResults || !this.lastResults.multiHandedness) {
            return [];
        }
        return this.lastResults.multiHandedness;
    }

    // Check if hands are detected
    hasHands() {
        return this.getHandLandmarks().length > 0;
    }

    // Get number of detected hands
    getHandCount() {
        return this.getHandLandmarks().length;
    }

    // Get confidence scores
    getConfidenceScores() {
        const handedness = this.getHandedness();
        return handedness.map(h => h.score);
    }

    // Simple gesture recognition (basic implementation)
    recognizeBasicGestures() {
        const landmarks = this.getHandLandmarks();
        if (landmarks.length === 0) return [];

        const gestures = [];
        
        for (let i = 0; i < landmarks.length; i++) {
            const hand = landmarks[i];
            const gesture = this.classifyHandGesture(hand);
            gestures.push(gesture);
        }

        return gestures;
    }

    classifyHandGesture(landmarks) {
        // Basic gesture classification
        // This is a simplified version - you'll expand this for ISL
        
        // Get key landmarks
        const thumb_tip = landmarks[4];
        const index_tip = landmarks[8];
        const middle_tip = landmarks[12];
        const ring_tip = landmarks[16];
        const pinky_tip = landmarks[20];
        
        const thumb_mcp = landmarks[2];
        const index_mcp = landmarks[5];
        const middle_mcp = landmarks[9];
        const ring_mcp = landmarks[13];
        const pinky_mcp = landmarks[17];

        // Simple finger counting logic
        let fingersUp = 0;
        
        // Thumb (different logic due to orientation)
        if (thumb_tip.x > thumb_mcp.x) fingersUp++;
        
        // Other fingers
        if (index_tip.y < index_mcp.y) fingersUp++;
        if (middle_tip.y < middle_mcp.y) fingersUp++;
        if (ring_tip.y < ring_mcp.y) fingersUp++;
        if (pinky_tip.y < pinky_mcp.y) fingersUp++;

        // Basic gesture mapping
        switch (fingersUp) {
            case 0: return { name: 'FIST', confidence: 0.8 };
            case 1: return { name: 'ONE', confidence: 0.8 };
            case 2: return { name: 'TWO', confidence: 0.8 };
            case 3: return { name: 'THREE', confidence: 0.8 };
            case 4: return { name: 'FOUR', confidence: 0.8 };
            case 5: return { name: 'FIVE', confidence: 0.8 };
            default: return { name: 'UNKNOWN', confidence: 0.5 };
        }
    }

    // Draw landmarks on canvas (for debugging)
    drawLandmarks(canvas, landmarks) {
        if (!canvas || !landmarks) return;

        const ctx = canvas.getContext('2d');
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;

        // Draw connections
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        
        // Hand connections (simplified)
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8], // Index
            [0, 9], [9, 10], [10, 11], [11, 12], // Middle
            [0, 13], [13, 14], [14, 15], [15, 16], // Ring
            [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
            [5, 9], [9, 13], [13, 17] // Palm connections
        ];

        connections.forEach(([start, end]) => {
            const startPoint = landmarks[start];
            const endPoint = landmarks[end];
            
            ctx.beginPath();
            ctx.moveTo(startPoint.x * canvasWidth, startPoint.y * canvasHeight);
            ctx.lineTo(endPoint.x * canvasWidth, endPoint.y * canvasHeight);
            ctx.stroke();
        });

        // Draw landmarks
        ctx.fillStyle = '#FF0000';
        landmarks.forEach(landmark => {
            ctx.beginPath();
            ctx.arc(
                landmark.x * canvasWidth,
                landmark.y * canvasHeight,
                5, 0, 2 * Math.PI
            );
            ctx.fill();
        });
    }

    // Cleanup
    destroy() {
        if (this.hands) {
            this.hands.close();
            this.hands = null;
        }
        this.isInitialized = false;
        this.lastResults = null;
        console.log('MediaPipe Hands processor destroyed');
    }
}

// Export for use in other modules
window.MediaPipeHandsProcessor = MediaPipeHandsProcessor;