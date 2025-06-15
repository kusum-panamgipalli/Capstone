// Video processing module for ISL Interpreter
// This module handles video stream capture and frame processing

class VideoProcessor {
    constructor() {
        this.canvas = null;
        this.context = null;
        this.isProcessing = false;
        this.frameCount = 0;
        this.lastFrameTime = 0;
        this.actualFPS = 0;
        this.isInitialized = false;
        
        // Processing settings
        this.targetWidth = 640;
        this.targetHeight = 480;
        this.flipHorizontal = true; // Mirror the video like a mirror
        
        // Initialize when DOM is ready
        this.initialize();
    }
    
    initialize() {
        try {
            // Check if document and body are ready
            if (!document || !document.body) {
                console.warn('Document not ready, retrying initialization...');
                setTimeout(() => this.initialize(), 100);
                return false;
            }
            
            // Remove existing canvas if it exists
            const existingCanvas = document.getElementById('isl-processing-canvas');
            if (existingCanvas) {
                existingCanvas.remove();
            }
            
            // Create hidden canvas for frame processing
            this.canvas = document.createElement('canvas');
            this.canvas.style.display = 'none';
            this.canvas.style.position = 'absolute';
            this.canvas.style.top = '-9999px';
            this.canvas.style.left = '-9999px';
            this.canvas.id = 'isl-processing-canvas';
            this.canvas.width = this.targetWidth;
            this.canvas.height = this.targetHeight;
            
            // Append to body
            document.body.appendChild(this.canvas);
            
            // Get context
            this.context = this.canvas.getContext('2d');
            
            if (!this.context) {
                throw new Error('Failed to get 2D context');
            }
            
            this.isInitialized = true;
            console.log('VideoProcessor initialized successfully');
            return true;
            
        } catch (error) {
            console.error('VideoProcessor initialization failed:', error);
            this.isInitialized = false;
            return false;
        }
    }
    
    // Check if processor is ready
    isReady() {
        return this.isInitialized && this.canvas && this.context;
    }
    
    // Main function to process a video frame
    processFrame(videoElement, callback) {
        if (!this.isReady()) {
            console.warn('VideoProcessor not ready, attempting to reinitialize...');
            if (!this.initialize()) {
                console.error('Failed to reinitialize VideoProcessor');
                return null;
            }
        }
        
        if (!videoElement || !this.isValidVideo(videoElement)) {
            console.warn('Invalid video element for processing');
            return null;
        }
        
        try {
            // Update FPS tracking
            this.updateFPSCounter();
            
            // Capture frame to canvas
            const frameData = this.captureFrame(videoElement);
            
            if (frameData) {
                // Process the frame data
                const processedData = this.preprocessFrame(frameData);
                
                // Call callback with processed data
                if (callback && typeof callback === 'function') {
                    callback(processedData);
                }
                
                return processedData;
            }
            
        } catch (error) {
            console.error('Error processing video frame:', error);
            return null;
        }
    }
    
    captureFrame(videoElement) {
        if (!this.isReady()) {
            console.error('VideoProcessor not ready for frame capture');
            return null;
        }
        
        const videoWidth = videoElement.videoWidth;
        const videoHeight = videoElement.videoHeight;
        
        if (videoWidth === 0 || videoHeight === 0) {
            return null;
        }
        
        try {
            // Calculate aspect-ratio-preserving dimensions
            const aspectRatio = videoWidth / videoHeight;
            let canvasWidth = this.targetWidth;
            let canvasHeight = this.targetHeight;
            
            if (aspectRatio > (this.targetWidth / this.targetHeight)) {
                canvasHeight = this.targetWidth / aspectRatio;
            } else {
                canvasWidth = this.targetHeight * aspectRatio;
            }
            
            // Set canvas size
            this.canvas.width = canvasWidth;
            this.canvas.height = canvasHeight;
            
            // Clear canvas
            this.context.clearRect(0, 0, canvasWidth, canvasHeight);
            
            // Save context for transformations
            this.context.save();
            
            // Flip horizontally if enabled (mirror effect)
            if (this.flipHorizontal) {
                this.context.scale(-1, 1);
                this.context.translate(-canvasWidth, 0);
            }
            
            // Draw video frame to canvas
            this.context.drawImage(
                videoElement,
                0, 0, videoWidth, videoHeight,  // Source rectangle
                0, 0, canvasWidth, canvasHeight  // Destination rectangle
            );
            
            // Restore context
            this.context.restore();
            
            // Get image data
            const imageData = this.context.getImageData(0, 0, canvasWidth, canvasHeight);
            
            return {
                imageData: imageData,
                width: canvasWidth,
                height: canvasHeight,
                timestamp: Date.now(),
                frameNumber: this.frameCount++
            };
            
        } catch (error) {
            console.error('Error capturing frame:', error);
            return null;
        }
    }
    
    preprocessFrame(frameData) {
        // This is where we'll add image preprocessing
        // For now, just return the frame data with some basic info
        
        return {
            ...frameData,
            processed: true,
            actualFPS: this.actualFPS,
            // Future: normalized data, noise reduction, etc.
        };
    }
    
    isValidVideo(videoElement) {
        return videoElement && 
               videoElement.videoWidth > 0 && 
               videoElement.videoHeight > 0 && 
               !videoElement.paused &&
               videoElement.readyState >= 2; // HAVE_CURRENT_DATA
    }
    
    updateFPSCounter() {
        const now = performance.now();
        if (this.lastFrameTime > 0) {
            const deltaTime = now - this.lastFrameTime;
            this.actualFPS = Math.round(1000 / deltaTime * 10) / 10; // Round to 1 decimal
        }
        this.lastFrameTime = now;
    }
    
    // Get current processing stats
    getStats() {
        return {
            frameCount: this.frameCount,
            actualFPS: this.actualFPS,
            canvasSize: {
                width: this.canvas?.width || 0,
                height: this.canvas?.height || 0
            },
            isProcessing: this.isProcessing,
            isInitialized: this.isInitialized,
            isReady: this.isReady()
        };
    }
    
    // Enable/disable processing
    setProcessing(enabled) {
        this.isProcessing = enabled;
        console.log(`Video processing ${enabled ? 'enabled' : 'disabled'}`);
    }
    
    // Cleanup resources
    destroy() {
        try {
            if (this.canvas) {
                this.canvas.remove();
                this.canvas = null;
                this.context = null;
            }
            this.isProcessing = false;
            this.isInitialized = false;
            console.log('VideoProcessor destroyed');
        } catch (error) {
            console.error('Error destroying VideoProcessor:', error);
        }
    }
    
    // Debug function to save current frame as image
    saveFrameForDebug(filename = 'debug-frame.png') {
        if (!this.canvas) {
            console.warn('No canvas available for debug save');
            return;
        }
        
        try {
            // Create download link
            this.canvas.toBlob(blob => {
                if (blob) {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    a.click();
                    URL.revokeObjectURL(url);
                } else {
                    console.error('Failed to create blob from canvas');
                }
            });
        } catch (error) {
            console.error('Error saving debug frame:', error);
        }
    }
    
    // Get canvas data URL for debugging
    getCanvasDataURL() {
        try {
            return this.canvas ? this.canvas.toDataURL() : null;
        } catch (error) {
            console.error('Error getting canvas data URL:', error);
            return null;
        }
    }
    
    // Manual reinitialize method
    reinitialize() {
        console.log('Manually reinitializing VideoProcessor...');
        this.destroy();
        return this.initialize();
    }
}