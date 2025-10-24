"""
Convert trained TensorFlow model to TensorFlow.js format
This allows the model to run in the browser extension
"""

import os
import json
import shutil
import tensorflowjs as tfjs

def convert_model_to_tfjs(model_dir='./models', output_dir='../isl-interpreter-extension/models'):
    """Convert TensorFlow SavedModel to TensorFlow.js format"""
    
    print("\n" + "="*60)
    print("CONVERTING MODEL TO TENSORFLOW.JS FORMAT")
    print("="*60)
    
    # Paths
    saved_model_path = os.path.join(model_dir, 'isl_model_saved')
    tfjs_output_path = os.path.abspath(output_dir)
    
    # Check if model exists
    if not os.path.exists(saved_model_path):
        print(f"\n❌ Model not found at: {saved_model_path}")
        print("\nPlease run 3_train_model.py first to train the model")
        return False
    
    # Create output directory
    os.makedirs(tfjs_output_path, exist_ok=True)
    
    print(f"\nInput model: {saved_model_path}")
    print(f"Output directory: {tfjs_output_path}")
    
    try:
        print("\nConverting model...")
        
        # Convert model
        tfjs.converters.convert_tf_saved_model(
            saved_model_path,
            tfjs_output_path,
            quantization_dtype_map={'float16': '*'}  # Quantize to reduce size
        )
        
        print("✓ Model converted successfully!")
        
        # Copy metadata and scaler
        copy_metadata_files(model_dir, tfjs_output_path)
        
        # List output files
        print("\nGenerated files:")
        for file in os.listdir(tfjs_output_path):
            file_path = os.path.join(tfjs_output_path, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  {file} ({file_size:.2f} KB)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure tensorflowjs is installed: pip install tensorflowjs")
        print("2. Check that the model trained successfully")
        print("3. Verify TensorFlow version compatibility")
        return False

def copy_metadata_files(source_dir, dest_dir):
    """Copy metadata and scaler files"""
    print("\nCopying metadata files...")
    
    files_to_copy = [
        'model_metadata.json',
        'scaler.pkl'
    ]
    
    for file in files_to_copy:
        source = os.path.join(source_dir, file)
        dest = os.path.join(dest_dir, file)
        
        if os.path.exists(source):
            shutil.copy2(source, dest)
            print(f"  ✓ Copied: {file}")
        else:
            print(f"  ⚠ Not found: {file}")
    
    # Create JavaScript-friendly label mapping
    create_js_label_mapping(source_dir, dest_dir)

def create_js_label_mapping(source_dir, dest_dir):
    """Create a JavaScript-friendly version of label mapping"""
    
    metadata_path = os.path.join(source_dir, 'model_metadata.json')
    
    if not os.path.exists(metadata_path):
        print("  ⚠ model_metadata.json not found")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create JS file with label mappings
    js_content = f"""// ISL Model Configuration
// Auto-generated from Python training script

const ISL_MODEL_CONFIG = {{
    numClasses: {metadata['num_classes']},
    inputShape: {metadata['input_shape']},
    labels: {json.dumps(list(metadata['label_to_idx'].keys()))},
    
    labelToIdx: {json.dumps(metadata['label_to_idx'])},
    
    idxToLabel: {json.dumps({str(k): v for k, v in metadata['idx_to_label'].items()})},
    
    modelInfo: {{
        architecture: "{metadata['model_architecture']}",
        featureType: "{metadata['feature_type']}",
        inputFeatures: {metadata['input_shape']}
    }}
}};

// Export for use in extension
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = ISL_MODEL_CONFIG;
}}

if (typeof window !== 'undefined') {{
    window.ISL_MODEL_CONFIG = ISL_MODEL_CONFIG;
}}
"""
    
    js_path = os.path.join(dest_dir, 'model-config.js')
    with open(js_path, 'w') as f:
        f.write(js_content)
    
    print(f"  ✓ Created: model-config.js")

def create_test_html(output_dir):
    """Create a simple HTML file to test the model in browser"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>ISL Model Test</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0"></script>
    <script src="model-config.js"></script>
</head>
<body>
    <h1>ISL Model Test</h1>
    <button onclick="loadModel()">Load Model</button>
    <button onclick="testPrediction()">Test Prediction</button>
    <div id="output"></div>
    
    <script>
        let model = null;
        
        async function loadModel() {
            const output = document.getElementById('output');
            output.innerHTML = 'Loading model...';
            
            try {
                model = await tf.loadLayersModel('./model.json');
                output.innerHTML = '✓ Model loaded successfully!<br>';
                output.innerHTML += `Input shape: ${model.inputs[0].shape}<br>`;
                output.innerHTML += `Output shape: ${model.outputs[0].shape}<br>`;
                output.innerHTML += `Classes: ${ISL_MODEL_CONFIG.numClasses}`;
            } catch (error) {
                output.innerHTML = '❌ Error loading model: ' + error.message;
            }
        }
        
        async function testPrediction() {
            const output = document.getElementById('output');
            
            if (!model) {
                output.innerHTML = 'Please load model first';
                return;
            }
            
            try {
                // Create random input (63 features for hand landmarks)
                const input = tf.randomNormal([1, ISL_MODEL_CONFIG.inputShape]);
                
                output.innerHTML = 'Making prediction...<br>';
                
                const prediction = model.predict(input);
                const probabilities = await prediction.data();
                const predictedClass = prediction.argMax(-1).dataSync()[0];
                
                output.innerHTML = `✓ Prediction complete!<br>`;
                output.innerHTML += `Predicted class: ${ISL_MODEL_CONFIG.idxToLabel[predictedClass]}<br>`;
                output.innerHTML += `Confidence: ${(probabilities[predictedClass] * 100).toFixed(2)}%<br>`;
                
                // Cleanup tensors
                input.dispose();
                prediction.dispose();
                
            } catch (error) {
                output.innerHTML = '❌ Error making prediction: ' + error.message;
            }
        }
    </script>
</body>
</html>
"""
    
    test_html_path = os.path.join(output_dir, 'test-model.html')
    with open(test_html_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✓ Created test file: test-model.html")
    print(f"  Open this file in a browser to test the model")

def main():
    print("\n" + "="*60)
    print("TENSORFLOWJS MODEL CONVERSION")
    print("="*60)
    
    success = convert_model_to_tfjs(
        model_dir='./models',
        output_dir='../isl-interpreter-extension/models'
    )
    
    if success:
        # Create test HTML file
        create_test_html('../isl-interpreter-extension/models')
        
        print("\n" + "="*60)
        print("CONVERSION COMPLETE!")
        print("="*60)
        print("\n✓ Model is ready for use in the Chrome extension")
        print("\nNext steps:")
        print("1. Load the extension in Chrome")
        print("2. Test on Google Meet/Zoom")
        print("3. Check model-config.js for label mappings")
    else:
        print("\n❌ Conversion failed. Please fix errors and try again.")

if __name__ == "__main__":
    main()
