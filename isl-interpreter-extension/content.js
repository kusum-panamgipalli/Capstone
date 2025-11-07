// Content script for ISL Interpreter - injected into video call pages

console.log('ISL Interpreter content script loaded');

// Global state
let isActive = false;
let settings = {
    showConfidence: true,
    processingSpeed: 20
};

let videoElement = null;
let processingInterval = null;
let overlayElement = null;
let videoProcessor = null;
let isVideoProcessorLoaded = false;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
} else {
    initialize();
}

async function initialize() {
    console.log('Initializing ISL Interpreter on:', window.location.href);
    
    // Inject library loader into page context (MediaPipe needs to run in page context)
    try {
        await injectLibraries();
        console.log('✓ Libraries injected successfully');
    } catch (error) {
        console.warn('Library injection failed, continuing anyway:', error);
    }
    
    // Load video processor first
    await loadVideoProcessor();
    
    // Set up message listener
    chrome.runtime.onMessage.addListener(handleMessage);
    
    // Start looking for video elements
    findVideoElements();
    
    // Set up mutation observer to detect dynamic video elements
    setupVideoObserver();
    
    // Notify background that we're ready
    sendToBackground('log', 'Content script initialized');
}

function handleMessage(message, sender, sendResponse) {
    console.log('Content script received message:', message);
    
    switch (message.action) {
        case 'ping':
            sendResponse({ status: 'alive' });
            break;
            
        case 'start':
            if (message.settings) {
                settings = { ...settings, ...message.settings };
            }
            startInterpreter();
            sendResponse({ status: 'started' });
            break;
            
        case 'stop':
            stopInterpreter();
            sendResponse({ status: 'stopped' });
            break;
            
        case 'updateSettings':
            if (message.settings) {
                settings = { ...settings, ...message.settings };
                if (isActive) {
                    updateProcessingSpeed();
                }
            }
            sendResponse({ status: 'settings_updated' });
            break;
            
        default:
            console.warn('Unknown message action:', message.action);
            sendResponse({ error: 'Unknown action' });
    }
}

function findVideoElements() {
    // Common selectors for video elements in different platforms
    const videoSelectors = [
        'video[autoplay]',  // General autoplay videos
        'video[src]',       // Videos with src
        '[data-participant-id] video',  // Google Meet specific
        '.participant-video video',     // Zoom specific
        '#local-video',     // Common local video ID
        '[class*="video"] video'  // Any element with "video" in class containing video
    ];
    
    for (const selector of videoSelectors) {
        const videos = document.querySelectorAll(selector);
        if (videos.length > 0) {
            // Prefer the first video that's actually playing
            for (const video of videos) {
                if (video.videoWidth > 0 && video.videoHeight > 0) {
                    console.log('Found active video element:', video);
                    setVideoElement(video);
                    return;
                }
            }
            
            // Fallback to first video found
            console.log('Found video element (not yet active):', videos[0]);
            setVideoElement(videos[0]);
            return;
        }
    }
    
    console.log('No video elements found, will retry...');
    setTimeout(findVideoElements, 2000); // Retry in 2 seconds
}

function setVideoElement(video) {
    if (videoElement === video) return;
    
    videoElement = video;
    
    // Notify popup that video was detected
    sendToPopup('videoDetected');
    
    console.log('Video element set:', {
        width: video.videoWidth,
        height: video.videoHeight,
        playing: !video.paused,
        muted: video.muted
    });
    
    // Set up video event listeners
    setupVideoEventListeners();
}

function setupVideoEventListeners() {
    if (!videoElement) return;
    
    videoElement.addEventListener('loadedmetadata', () => {
        console.log('Video metadata loaded:', {
            width: videoElement.videoWidth,
            height: videoElement.videoHeight
        });
        sendToPopup('videoDetected');
    });
    
    videoElement.addEventListener('play', () => {
        console.log('Video started playing');
        sendToPopup('videoDetected');
    });
    
    videoElement.addEventListener('pause', () => {
        console.log('Video paused');
    });
    
    videoElement.addEventListener('ended', () => {
        console.log('Video ended');
        sendToPopup('videoLost');
    });
}

function setupVideoObserver() {
    const observer = new MutationObserver((mutations) => {
        let shouldCheckForVideo = false;
        
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    // Check if the added node is a video or contains videos
                    if (node.tagName === 'VIDEO' || node.querySelector('video')) {
                        shouldCheckForVideo = true;
                    }
                }
            });
        });
        
        if (shouldCheckForVideo) {
            console.log('DOM mutation detected, checking for new video elements');
            setTimeout(findVideoElements, 500); // Small delay to let DOM settle
        }
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    console.log('Video observer set up');
}

function startInterpreter() {
    if (isActive) {
        console.log('Interpreter already active');
        return;
    }
    
    if (!videoElement) {
        console.warn('No video element found, cannot start interpreter');
        sendToBackground('error', 'No video element available');
        return;
    }
    
    if (!isVideoProcessorLoaded) {
        console.warn('Video processor not loaded, cannot start interpreter');
        sendToBackground('error', 'Video processor not available');
        return;
    }
    
    isActive = true;
    console.log('Starting ISL interpreter');
    
    // Create overlay UI
    createOverlay();
    
    // Start processing loop
    startProcessing();
    
    sendToBackground('log', 'Interpreter started successfully');
}

function stopInterpreter() {
    if (!isActive) return;
    
    isActive = false;
    console.log('Stopping ISL interpreter');
    
    // Stop processing
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    
    // Stop video processor
    if (videoProcessor) {
        videoProcessor.setProcessing(false);
    }
    
    // Remove overlay
    removeOverlay();
    
    sendToBackground('log', 'Interpreter stopped');
}

function createOverlay() {
    if (overlayElement) {
        removeOverlay();
    }
    
    overlayElement = document.createElement('div');
    overlayElement.id = 'isl-interpreter-overlay';
    overlayElement.innerHTML = `
        <div class="isl-header">
            <span>ISL Interpreter</span>
            <button class="isl-close" onclick="window.islInterpreter.stop()">×</button>
        </div>
        <div class="isl-output">
            <div class="isl-text">Ready to interpret...</div>
            <div class="isl-confidence" style="display: ${settings.showConfidence ? 'block' : 'none'}">
                Confidence: --
            </div>
        </div>
    `;
    
    // Add CSS styles
    const style = document.createElement('style');
    style.textContent = `
        #isl-interpreter-overlay {
            position: fixed;
            top: 20px;
            right: 20px;
            width: 300px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 10px;
            padding: 0;
            font-family: Arial, sans-serif;
            z-index: 10000;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            border: 2px solid #4CAF50;
        }
        
        .isl-header {
            background: #4CAF50;
            padding: 10px 15px;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: bold;
        }
        
        .isl-close {
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            width: 25px;
            height: 25px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .isl-close:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .isl-output {
            padding: 15px;
        }
        
        .isl-text {
            font-size: 16px;
            margin-bottom: 10px;
            min-height: 20px;
        }
        
        .isl-confidence {
            font-size: 12px;
            color: #ccc;
        }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(overlayElement);
    
    // Make it globally accessible for the close button
    window.islInterpreter = {
        stop: () => {
            sendToPopup('stop');
            stopInterpreter();
        }
    };
    
    console.log('Overlay created');
}

function removeOverlay() {
    if (overlayElement) {
        overlayElement.remove();
        overlayElement = null;
    }
    
    // Clean up global reference
    if (window.islInterpreter) {
        delete window.islInterpreter;
    }
}

function startProcessing() {
    const fps = settings.processingSpeed;
    const intervalMs = 1000 / fps;
    
    console.log(`Starting processing at ${fps} FPS (${intervalMs}ms interval)`);
    
    processingInterval = setInterval(() => {
        if (!isActive || !videoElement) {
            stopInterpreter();
            return;
        }
        
        processVideoFrame();
    }, intervalMs);
}

function updateProcessingSpeed() {
    if (processingInterval) {
        clearInterval(processingInterval);
        startProcessing();
    }
}

function processVideoFrame() {
    try {
        if (!videoProcessor) {
            console.warn('Video processor not initialized');
            return;
        }
        
        // Process the current video frame - let the VideoProcessor handle canvas checks internally
        const frameData = videoProcessor.processFrame(videoElement, (processedData) => {
            // This callback will be called with processed frame data
            handleProcessedFrame(processedData);
        });
        
        if (frameData) {
            // Update overlay with processing stats
            const stats = videoProcessor.getStats();
            updateOverlayText(
                `Processing... Frame ${stats.frameCount}`, 
                `${stats.actualFPS} FPS`
            );
        } else {
            // If no frame data, show status
            updateOverlayText('Waiting for video...', 'No data');
        }
        
    } catch (error) {
        console.error('Error processing video frame:', error);
        sendToBackground('error', error.message);
    }
}

function handleProcessedFrame(processedData) {
    // Enhanced processing with MediaPipe integration
    
    if (processedData.frameNumber % 30 === 0) { // Log every 30 frames
        console.log('Processed frame:', {
            frameNumber: processedData.frameNumber,
            dimensions: `${processedData.width}x${processedData.height}`,
            timestamp: processedData.timestamp,
            fps: processedData.actualFPS,
            hasMediaPipe: processedData.hasMediaPipe,
            handCount: processedData.handCount,
            gestures: processedData.gestures
        });
    }

    // Update overlay with MediaPipe results
    if (processedData.hasMediaPipe) {
        if (processedData.handCount > 0) {
            // Display detected gestures
            const gestureText = processedData.gestureText || 'Processing...';
            const confidenceText = processedData.confidenceScores.length > 0 
                ? `Avg: ${Math.round(processedData.confidenceScores.reduce((a, b) => a + b, 0) / processedData.confidenceScores.length * 100)}%`
                : 'No confidence';
            
            updateOverlayText(gestureText, confidenceText);
        } else {
            // No hands detected
            updateOverlayText('No hands detected', 'Position hands in view');
        }
    } else {
        // MediaPipe not ready or failed
        updateOverlayText('Loading MediaPipe...', 'Please wait');
    }
}

// Add this debug function to your content.js
function debugMediaPipe() {
    if (window.islDebug && window.islDebug.videoProcessor) {
        const processor = window.islDebug.videoProcessor();
        const stats = processor.getStats();
        
        console.log('MediaPipe Debug Info:', {
            isMediaPipeReady: stats.isMediaPipeReady,
            handCount: stats.handCount,
            lastGestures: stats.lastGestures,
            frameCount: stats.frameCount,
            actualFPS: stats.actualFPS
        });
        
        return stats;
    }
    return null;
}


async function injectLibraries() {
    console.log('Injecting external libraries into page context...');
    
    return new Promise((resolve, reject) => {
        try {
            // Create script element to inject into page
            const script = document.createElement('script');
            script.src = chrome.runtime.getURL('injector.js');
            script.onload = () => {
                console.log('✓ Injector script loaded');
                
                // Listen for library load confirmation
                const listener = (event) => {
                    if (event.data && event.data.type === 'ISL_LIBRARIES_LOADED') {
                        console.log('✓ External libraries loaded successfully');
                        window.removeEventListener('message', listener);
                        // Wait a bit to ensure everything is ready
                        setTimeout(resolve, 500);
                    } else if (event.data && event.data.type === 'ISL_LIBRARIES_ERROR') {
                        console.error('Failed to load libraries:', event.data.error);
                        window.removeEventListener('message', listener);
                        reject(new Error(event.data.error));
                    }
                };
                
                window.addEventListener('message', listener);
                
                // Timeout after 10 seconds (reduced from 30)
                setTimeout(() => {
                    window.removeEventListener('message', listener);
                    console.warn('Library loading timeout - proceeding anyway');
                    resolve();
                }, 10000);
            };
            
            script.onerror = () => {
                console.error('Failed to inject library loader');
                reject(new Error('Failed to inject library loader'));
            };
            
            (document.head || document.documentElement).appendChild(script);
        } catch (error) {
            console.error('Error injecting libraries:', error);
            reject(error);
        }
    });
}


async function loadVideoProcessor() {
    try {
        console.log('Loading video processor...');
        
        // First, let's try to load it as a simple script injection
        const response = await fetch(chrome.runtime.getURL('video-processor.js'));
        const scriptContent = await response.text();
        
        // Create and inject the script
        const script = document.createElement('script');
        script.textContent = scriptContent;
        script.type = 'text/javascript';
        document.head.appendChild(script);
        
        // Wait a moment for the class to be available
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // Check if VideoProcessor class is available
        if (typeof VideoProcessor === 'undefined') {
            throw new Error('VideoProcessor class not found after script injection');
        }
        
        // Initialize video processor with retry logic
        let initAttempts = 0;
        const maxAttempts = 5;
        
        while (initAttempts < maxAttempts) {
            try {
                console.log(`Attempting to initialize VideoProcessor (attempt ${initAttempts + 1}/${maxAttempts})`);
                videoProcessor = new VideoProcessor();
                
                // Wait for initialization to complete
                await new Promise(resolve => setTimeout(resolve, 100));
                
                // Check if it's properly initialized
                if (videoProcessor.isReady && videoProcessor.isReady()) {
                    isVideoProcessorLoaded = true;
                    console.log('Video processor loaded and initialized successfully');
                    return;
                }
                
                console.warn('VideoProcessor created but not ready, retrying...');
                initAttempts++;
                await new Promise(resolve => setTimeout(resolve, 500));
                
            } catch (initError) {
                console.warn(`Initialization attempt ${initAttempts + 1} failed:`, initError);
                initAttempts++;
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        }
        
        throw new Error(`Failed to initialize VideoProcessor after ${maxAttempts} attempts`);
        
    } catch (error) {
        console.error('Failed to load video processor:', error);
        isVideoProcessorLoaded = false;
        videoProcessor = null;
        
        // Create a fallback processor
        createFallbackProcessor();
    }
}

function createFallbackProcessor() {
    console.log('Creating fallback video processor');
    
    videoProcessor = {
        processFrame: (videoElement, callback) => {
            console.log('Using fallback processor');
            const fallbackData = {
                frameNumber: Date.now(),
                width: videoElement.videoWidth || 640,
                height: videoElement.videoHeight || 480,
                timestamp: Date.now(),
                actualFPS: 0
            };
            
            if (callback) {
                callback(fallbackData);
            }
            
            return fallbackData;
        },
        
        getStats: () => ({
            frameCount: 0,
            actualFPS: 0,
            canvasSize: { width: 0, height: 0 },
            isProcessing: false
        }),
        
        setProcessing: (enabled) => {
            console.log(`Fallback processor ${enabled ? 'enabled' : 'disabled'}`);
        }
    };
    
    isVideoProcessorLoaded = true;
}

function updateOverlayText(text, confidence) {
    if (!overlayElement) return;
    
    const textElement = overlayElement.querySelector('.isl-text');
    const confidenceElement = overlayElement.querySelector('.isl-confidence');
    
    if (textElement) {
        textElement.textContent = text;
    }
    
    if (confidenceElement && settings.showConfidence) {
        confidenceElement.textContent = `Confidence: ${confidence}`;
        confidenceElement.style.display = 'block';
    } else if (confidenceElement) {
        confidenceElement.style.display = 'none';
    }
}

// Utility functions
function sendToBackground(action, data) {
    chrome.runtime.sendMessage({ action, data }).catch(console.error);
}

function sendToPopup(action, data) {
    chrome.runtime.sendMessage({ action, data }).catch(console.error);
}

// Export for debugging
if (typeof window !== 'undefined'&& window.islDebug) {
    window.islDebug = {
        debugMediaPipe: debugMediaPipe,
        toggleLandmarks : () => {
            if (videoProcessor && videoProcessor.toggleLandmarks) {
                videoProcessor.toggleLandmarks();
                }
            },
        isActive: () => isActive,
        videoElement: () => videoElement,
        videoProcessor: () => videoProcessor,
        isVideoProcessorLoaded: () => isVideoProcessorLoaded,
        settings: () => settings,
        findVideoElements,
        startInterpreter,
        stopInterpreter,
        loadVideoProcessor,
        checkCanvas: () => {
            const canvas = document.getElementById('isl-processing-canvas');
            console.log('Canvas check:', {
                exists: !!canvas,
                element: canvas,
                processorReady: videoProcessor?.isReady?.() || false,
                processorStats: videoProcessor?.getStats?.() || null
            });
            return canvas;
        },
        reinitializeProcessor: () => {
            if (videoProcessor && videoProcessor.reinitialize) {
                return videoProcessor.reinitialize();
            }
            return false;
        }
    };
}