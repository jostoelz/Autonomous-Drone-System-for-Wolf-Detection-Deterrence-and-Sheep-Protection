import cv2
import numpy as np
import onnxruntime as ort
import time

# --- CONFIGURATION ---
MODEL_PATH = "depth_anything_v2_metric_outdoor.onnx" 
IMAGE_PATH = "test_bild.jpg"                         # test image
INPUT_SIZE = (518, 518)                              # standard for Depth Anything

def run_inference():
    # start ONNX Runtime Session (CPU)
    print(f"Loading model: {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    
    # finds the name of the input layer
    input_name = session.get_inputs()[0].name
    
    # loads and prepares the image 
    image = cv2.imread(IMAGE_PATH)

    h_orig, w_orig = image.shape[:2] # saves original dimensions for later
    
    # resizes to model input size
    image_resized = cv2.resize(image, INPUT_SIZE)
    
    # openCV loads BGR, model needs RGB -> Convert
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # normalization (Standard ImageNet values)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_rgb / 255.0 - mean) / std
    
    # transposes from HWC (Height, Width, Channel) to CHW (Channel, Height, Width)
    image_input = image_norm.transpose(2, 0, 1).astype(np.float32)
    
    # adds batch dimension (1, 3, 518, 518)
    image_input = np.expand_dims(image_input, axis=0)

    # performs inference
    start_time = time.time()
    
    outputs = session.run(None, {input_name: image_input})
    
    end_time = time.time()
    print(f"Inference Time: {end_time - start_time:.4f} seconds")

    # process Result 
    depth = outputs[0] # raw data
    
    # cleans up dimensions
    if len(depth.shape) == 3:
        depth = depth[0] # If (1, H, W)
        
    # scales back to original size
    depth = cv2.resize(depth, (w_orig, h_orig))

    # normalizes to 0-255 
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
  
    # applys color map 
    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    # saves
    cv2.imwrite("output_depth.jpg", depth_color)
    print("Result saved as 'output_depth.jpg'")

if __name__ == "__main__":
    run_inference()
