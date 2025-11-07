// Popup script for ISL Interpreter Extension

let isActive = false;
let currentTab = null;

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', initialize);

async function initialize() {
    console.log('ISL Interpreter popup initialized');
    
    // Get current tab info
    await getCurrentTabInfo();
    
    // Load saved settings
    await loadSettings();
    
    // Set up event listeners
    setupEventListeners();
    
    // Update UI state
    updateUI();
}

async function getCurrentTabInfo() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        currentTab = tab;
        
        // Update URL display
        const urlElement = document.getElementById('currentUrl');
        urlElement.textContent = tab.url || 'Unknown';
        
        // Check if we're on a supported video call platform
        const isSupportedSite = tab.url.includes('meet.google.com') || 
                               tab.url.includes('zoom.us');
        
        if (!isSupportedSite) {
            document.getElementById('status').textContent = 'Unsupported site';
            document.getElementById('toggleButton').disabled = true;
        }
        
    } catch (error) {
        console.error('Error getting tab info:', error);
    }
}

async function loadSettings() {
    try {
        const result = await chrome.storage.sync.get({
            isActive: false,
            showConfidence: true,
            processingSpeed: 20
        });
        
        isActive = result.isActive;
        document.getElementById('showConfidence').checked = result.showConfidence;
        document.getElementById('processingSpeed').value = result.processingSpeed;
        
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

function setupEventListeners() {
    // Toggle button
    document.getElementById('toggleButton').addEventListener('click', toggleInterpreter);
    
    // Settings changes
    document.getElementById('showConfidence').addEventListener('change', saveSettings);
    document.getElementById('processingSpeed').addEventListener('change', saveSettings);
    
    // Listen for messages from content script
    chrome.runtime.onMessage.addListener(handleMessage);
}

async function toggleInterpreter() {
    try {
        isActive = !isActive;
        
        // Save state
        await chrome.storage.sync.set({ isActive });
        
        // Send message to content script
        if (currentTab) {
            await chrome.tabs.sendMessage(currentTab.id, {
                action: isActive ? 'start' : 'stop',
                settings: await getSettings()
            });
        }
        
        updateUI();
        
    } catch (error) {
        console.error('Error toggling interpreter:', error);
        // Reset state on error
        isActive = false;
        updateUI();
    }
}

async function saveSettings() {
    try {
        const settings = await getSettings();
        await chrome.storage.sync.set(settings);
        
        // Send updated settings to content script if active
        if (isActive && currentTab) {
            await chrome.tabs.sendMessage(currentTab.id, {
                action: 'updateSettings',
                settings: settings
            });
        }
        
    } catch (error) {
        console.error('Error saving settings:', error);
    }
}

async function getSettings() {
    return {
        isActive,
        showConfidence: document.getElementById('showConfidence').checked,
        processingSpeed: parseInt(document.getElementById('processingSpeed').value)
    };
}

function updateUI() {
    const statusElement = document.getElementById('status');
    const toggleButton = document.getElementById('toggleButton');
    
    if (isActive) {
        statusElement.textContent = 'Active';
        statusElement.style.background = 'rgba(76, 175, 80, 0.8)';
        toggleButton.textContent = 'Disable Interpreter';
        toggleButton.classList.add('active');
    } else {
        statusElement.textContent = 'Inactive';
        statusElement.style.background = 'rgba(244, 67, 54, 0.8)';
        toggleButton.textContent = 'Enable Interpreter';
        toggleButton.classList.remove('active');
    }
}

function handleMessage(message, sender, sendResponse) {
    console.log('Popup received message:', message);
    
    switch (message.action) {
        case 'videoDetected':
            document.getElementById('videoStatus').textContent = 'Yes';
            break;
        case 'videoLost':
            document.getElementById('videoStatus').textContent = 'No';
            break;
        case 'error':
            console.error('Content script error:', message.error);
            // Auto-disable on error
            isActive = false;
            updateUI();
            break;
    }
    
    sendResponse({ received: true });
}