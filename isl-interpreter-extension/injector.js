// Script injector to load external libraries in page context
// This runs in the page context, not extension context, so it can load external scripts

(function() {
    'use strict';
    
    console.log('ISL Injector: Loading required libraries...');
    
    // Function to load script dynamically
    function loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = () => {
                console.log(`✓ Loaded: ${src}`);
                resolve();
            };
            script.onerror = () => {
                console.error(`✗ Failed to load: ${src}`);
                reject(new Error(`Failed to load ${src}`));
            };
            document.head.appendChild(script);
        });
    }
    
    // Load required libraries
    async function loadLibraries() {
        try {
            console.log('Starting to load MediaPipe libraries...');
            
            // Load MediaPipe dependencies only (no TensorFlow.js needed - using Flask API)
            await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675465747/camera_utils.js');
            console.log('✓ Camera utils loaded');
            
            await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1675465747/drawing_utils.js');
            console.log('✓ Drawing utils loaded');
            
            await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.js');
            console.log('✓ Hands loaded');
            
            console.log('✓ MediaPipe libraries loaded');
            
            // Signal that libraries are ready
            window.postMessage({ type: 'ISL_LIBRARIES_LOADED' }, '*');
            console.log('✓ All ISL libraries loaded successfully');
            
        } catch (error) {
            console.error('Failed to load ISL libraries:', error);
            window.postMessage({ type: 'ISL_LIBRARIES_ERROR', error: error.message }, '*');
        }
    }
    
    // Start loading
    loadLibraries();
})();
