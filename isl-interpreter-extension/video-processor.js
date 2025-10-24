// Video Processor with MediaPipe and TensorFlow.js Integration for ISL Interpreter

class VideoProcessor {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.isProcessing = false;
        this.frameCount = 0;
        this.lastProcessTime = Date.now();
        this.fps = 0;
        
        // MediaPipe Hands
        this.hands = null;
        this.isMediaPipeReady = false;
        this.isMediaPipeLoading = false;
        this.lastResults = null;
        
        // TensorFlow.js Model
        this.model = null;
        this.isModelReady = false;
        this.isModelLoading = false;
        this.modelConfig = null;
        
        // MediaPipe configuration
        this.mediaPipeConfig = {
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
            maxNumHands: 1  // ISL typically uses one hand at a time
        };
        
        // Initialize automatically
        this.initialize();
    }

    async initialize() {
        console.log('Initializing VideoProcessor...');
        
        // Create processing canvas
        this.createCanvas();
        
        // Load MediaPipe (API-based, no need to load TF.js model)
        try {
            await this.initializeMediaPipe();
            console.log('✓ VideoProcessor initialized successfully');
            console.log('✓ Using Flask API for predictions (http://localhost:5000)');
            return true;
        } catch (error) {
            console.error('Failed to initialize VideoProcessor:', error);
            return false;
        }
    }
    
    createCanvas() {
        // Create hidden canvas for processing
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'isl-processing-canvas';
        this.canvas.style.display = 'none';
        document.body.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
        console.log('✓ Processing canvas created');
    }
    
    async initializeMediaPipe() {
        if (this.isMediaPipeReady || this.isMediaPipeLoading) {
            console.log(`MediaPipe status: ready=${this.isMediaPipeReady}, loading=${this.isMediaPipeLoading}`);
            return this.isMediaPipeReady;
        }

        this.isMediaPipeLoading = true;
        console.log('Loading MediaPipe Hands...');

        try {
            // Load MediaPipe scripts
            console.log('Attempting to load MediaPipe scripts...');
            await this.loadMediaPipeScripts();
            console.log('✓ MediaPipe scripts loaded');

            // Check if Hands is available
            if (typeof Hands === 'undefined') {
                throw new Error('MediaPipe Hands not available after loading scripts');
            }
            
            console.log('✓ MediaPipe Hands class is available');

            // Initialize Hands
            this.hands = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`;
                }
            });

            // Configure hands
            await this.hands.setOptions(this.mediaPipeConfig);

            // Set up results callback
            this.hands.onResults((results) => {
                this.lastResults = results;
            });

            this.isMediaPipeReady = true;
            console.log('✓ MediaPipe Hands initialized');
            return true;

        } catch (error) {
            console.error('Failed to initialize MediaPipe Hands:', error);
            this.isMediaPipeReady = false;
            return false;
        } finally {
            this.isMediaPipeLoading = false;
        }
    }
    
    async loadModel() {
        if (this.isModelReady || this.isModelLoading) {
            return this.isModelReady;
        }

        this.isModelLoading = true;
        console.log('Loading TensorFlow.js model...');

        try {
            // Load TensorFlow.js
            if (typeof tf === 'undefined') {
                await this.loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0');
            }

            // Load model configuration
            const configResponse = await fetch(chrome.runtime.getURL('models/model-config.js'));
            const configText = await configResponse.text();
            eval(configText);  // Loads ISL_MODEL_CONFIG
            this.modelConfig = window.ISL_MODEL_CONFIG;

            // Load the model
            const modelPath = chrome.runtime.getURL('models/model.json');
            this.model = await tf.loadLayersModel(modelPath);

            this.isModelReady = true;
            console.log('✓ TensorFlow.js model loaded');
            console.log(`  Classes: ${this.modelConfig.numClasses}`);
            console.log(`  Input shape: ${this.modelConfig.inputShape}`);
            
            return true;

        } catch (error) {
            console.error('Failed to load TensorFlow.js model:', error);
            console.warn('Extension will work with basic gesture recognition only');
            this.isModelReady = false;
            return false;
        } finally {
            this.isModelLoading = false;
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
                console.log(`Script already loaded: ${src}`);
                resolve();
                return;
            }

            console.log(`Loading script: ${src}`);
            const script = document.createElement('script');
            script.src = src;
            script.onload = () => {
                console.log(`✓ Script loaded: ${src}`);
                resolve();
            };
            script.onerror = (error) => {
                console.error(`✗ Failed to load script: ${src}`, error);
                reject(error);
            };
            document.head.appendChild(script);
        });
    }

    async processFrame(videoElement, callback) {
        if (!this.canvas || !this.ctx) {
            console.warn('Canvas not initialized');
            return null;
        }

        if (!videoElement || videoElement.videoWidth === 0) {
            return null;
        }

        try {
            // Update frame counter
            this.frameCount++;
            
            // Calculate FPS
            const now = Date.now();
            const elapsed = now - this.lastProcessTime;
            if (elapsed >= 1000) {
                this.fps = Math.round((this.frameCount * 1000) / elapsed);
                this.frameCount = 0;
                this.lastProcessTime = now;
            }

            // Update canvas size if needed
            if (this.canvas.width !== videoElement.videoWidth || 
                this.canvas.height !== videoElement.videoHeight) {
                this.canvas.width = videoElement.videoWidth;
                this.canvas.height = videoElement.videoHeight;
            }

            // Draw video frame to canvas
            this.ctx.drawImage(videoElement, 0, 0, this.canvas.width, this.canvas.height);

            // Process with MediaPipe if ready
            let processedData = {
                frameNumber: this.frameCount,
                width: this.canvas.width,
                height: this.canvas.height,
                timestamp: now,
                actualFPS: this.fps,
                hasMediaPipe: false,
                handCount: 0,
                gestures: [],
                gestureText: '',
                confidenceScores: []
            };

            if (this.isMediaPipeReady && this.hands) {
                await this.hands.send({ image: this.canvas });
                
                if (this.lastResults && this.lastResults.multiHandLandmarks) {
                    processedData.hasMediaPipe = true;
                    processedData.handCount = this.lastResults.multiHandLandmarks.length;
                    
                    // Recognize gesture using trained model
                    if (processedData.handCount > 0) {
                        const recognition = await this.recognizeGesture(this.lastResults.multiHandLandmarks[0]);
                        processedData.gestures = [recognition];
                        processedData.gestureText = recognition.label;
                        processedData.confidenceScores = [recognition.confidence];
                    }
                }
            }

            // Call callback with processed data
            if (callback) {
                callback(processedData);
            }

            return processedData;

        } catch (error) {
            console.error('Error processing frame:', error);
            return null;
        }
    }
    
    async recognizeGesture(landmarks) {
        // Use Flask API for prediction with trained model
        try {
            // Extract landmark features (flatten x, y, z coordinates)
            const landmarksArray = [];
            for (let i = 0; i < landmarks.length; i++) {
                landmarksArray.push([landmarks[i].x, landmarks[i].y, landmarks[i].z]);
            }
            
            // Call Flask API
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    landmarks: landmarksArray
                })
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Return result in expected format
            return {
                label: result.sign,
                confidence: result.confidence,
                allProbabilities: result.all_predictions
            };
            
        } catch (error) {
            // If API is not available or fails, fall back to basic recognition
            if (error.message.includes('fetch')) {
                console.warn('Flask API not available. Make sure to run: python api_server.py');
            } else {
                console.error('Error calling prediction API:', error);
            }
            // Fall back to basic gesture recognition
            return this.basicGestureRecognition(landmarks);
        }
    }
    
    basicGestureRecognition(landmarks) {
        // Simple finger counting for basic recognition
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

        let fingersUp = 0;
        
        // Thumb (different logic due to orientation)
        if (thumb_tip.x > thumb_mcp.x) fingersUp++;
        
        // Other fingers
        if (index_tip.y < index_mcp.y) fingersUp++;
        if (middle_tip.y < middle_mcp.y) fingersUp++;
        if (ring_tip.y < ring_mcp.y) fingersUp++;
        if (pinky_tip.y < pinky_mcp.y) fingersUp++;

        // Basic mapping
        const gestureMap = {
            0: 'Fist / A',
            1: 'One / B',
            2: 'Two / V',
            3: 'Three / W',
            4: 'Four',
            5: 'Five / Hand'
        };

        return {
            label: gestureMap[fingersUp] || 'Unknown',
            confidence: 0.7,
            allProbabilities: []
        };
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

    hasHands() {
        return this.getHandLandmarks().length > 0;
    }

    getHandCount() {
        return this.getHandLandmarks().length;
    }

    setProcessing(enabled) {
        this.isProcessing = enabled;
        console.log(`Processing ${enabled ? 'enabled' : 'disabled'}`);
    }

    getStats() {
        return {
            frameCount: this.frameCount,
            actualFPS: this.fps,
            canvasSize: { 
                width: this.canvas?.width || 0, 
                height: this.canvas?.height || 0 
            },
            isProcessing: this.isProcessing,
            isMediaPipeReady: this.isMediaPipeReady,
            isModelReady: this.isModelReady,
            handCount: this.getHandCount(),
            lastGestures: this.lastResults ? 'Available' : 'None'
        };
    }

    isReady() {
        // Consider ready if canvas exists (MediaPipe will load on-demand)
        return this.canvas !== null;
    }

    // Draw landmarks on canvas (for debugging)
    drawLandmarks(targetCanvas, landmarks) {
        if (!targetCanvas || !landmarks) return;

        const ctx = targetCanvas.getContext('2d');
        const canvasWidth = targetCanvas.width;
        const canvasHeight = targetCanvas.height;

        // Draw connections
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        
        // Hand connections
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

    toggleLandmarks() {
        // This can be called to show/hide landmarks overlay
        console.log('Toggle landmarks visualization');
    }

    reinitialize() {
        console.log('Reinitializing VideoProcessor...');
        this.cleanup();
        return this.initialize();
    }

    cleanup() {
        // Cleanup MediaPipe
        if (this.hands) {
            this.hands.close();
            this.hands = null;
        }
        
        // Cleanup TensorFlow.js model
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        
        // Remove canvas
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
        
        this.isMediaPipeReady = false;
        this.isModelReady = false;
        this.lastResults = null;
        
        console.log('VideoProcessor cleaned up');
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.VideoProcessor = VideoProcessor;
}