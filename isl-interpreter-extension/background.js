// Background service worker for ISL Interpreter Extension

console.log('ISL Interpreter background script loaded');

// Extension installation/update handler
chrome.runtime.onInstalled.addListener((details) => {
    console.log('ISL Interpreter installed/updated:', details.reason);
    
    if (details.reason === 'install') {
        // Set default settings on first install
        chrome.storage.sync.set({
            isActive: false,
            showConfidence: true,
            processingSpeed: 20,
            installDate: Date.now()
        });
        
        console.log('Default settings initialized');
    }
});

// Handle messages from content scripts and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('Background received message:', message);
    
    switch (message.action) {
        case 'log':
            console.log('Content script log:', message.data);
            break;
            
        case 'error':
            console.error('Content script error:', message.error);
            // Could implement error reporting here
            break;
            
        case 'requestPermissions':
            handlePermissionRequest(message.permissions)
                .then(granted => sendResponse({ granted }))
                .catch(error => sendResponse({ error: error.message }));
            return true; // Keep message channel open for async response
            
        default:
            console.warn('Unknown message action:', message.action);
    }
});

// Handle permission requests
async function handlePermissionRequest(permissions) {
    try {
        const granted = await chrome.permissions.request({
            permissions: permissions
        });
        
        console.log('Permission request result:', granted);
        return granted;
        
    } catch (error) {
        console.error('Permission request failed:', error);
        throw error;
    }
}

// Tab update handler - useful for detecting navigation
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    // Only process complete page loads on supported sites
    if (changeInfo.status === 'complete' && tab.url) {
        const isSupportedSite = tab.url.includes('meet.google.com') || 
                               tab.url.includes('zoom.us');
        
        if (isSupportedSite) {
            console.log('Supported video call site loaded:', tab.url);
            
            // Inject content script if not already present
            injectContentScriptIfNeeded(tabId);
        }
    }
});

// Ensure content script is injected
async function injectContentScriptIfNeeded(tabId) {
    try {
        // Try to ping existing content script
        const response = await chrome.tabs.sendMessage(tabId, { action: 'ping' });
        console.log('Content script already present');
        
    } catch (error) {
        // Content script not present, inject it
        console.log('Injecting content script into tab:', tabId);
        
        try {
            await chrome.scripting.executeScript({
                target: { tabId },
                files: ['content.js']
            });
            
            console.log('Content script injected successfully');
            
        } catch (injectError) {
            console.error('Failed to inject content script:', injectError);
        }
    }
}

// Alarm handlers for periodic tasks (if needed)
chrome.alarms.onAlarm.addListener((alarm) => {
    console.log('Alarm triggered:', alarm.name);
    
    switch (alarm.name) {
        case 'cleanup':
            performCleanup();
            break;
    }
});

// Cleanup function
function performCleanup() {
    console.log('Performing background cleanup');
    
    // Clean up old storage data, logs, etc.
    // This could be useful for maintaining performance
}

// Set up periodic cleanup (optional)
chrome.alarms.create('cleanup', {
    delayInMinutes: 60, // Run every hour
    periodInMinutes: 60
});