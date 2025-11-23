import cv2
import numpy as np
import onnxruntime as ort
import time

# --- CONFIGURATION ---
MODEL_PATH = "depth_anything_v2_vits_518x518.onnx" 
IMAGE_PATH = "frame_10199_jpg.rf.04226956b262c747a3c05bad7f92a126.jpg"                      # test image
INPUT_SIZE = (518, 518)                      # standard for Depth Anything

POINT_X = 745  # Horizontal (Column)
POINT_Y = 1065  # Vertical (Row)

def run_inference():
    # start ONNX Runtime Session (CPU)
    print(f"Loading model: {MODEL_PATH}...")
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # finds the name of the input layer
    input_name = session.get_inputs()[0].name
    
    # loads and prepares the image 
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image at {IMAGE_PATH}")
        return

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
    print("Starting inference...")
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

    if 0 <= POINT_X < w_orig and 0 <= POINT_Y < h_orig:
        # access numpy array as [row, column] -> [y, x]
        depth_value = depth[POINT_Y, POINT_X]
        print(f"Point Coordinates: X={POINT_X}, Y={POINT_Y}")
        print(f"Depth Value: {depth_value:.4f}") 
    else:
        print(f"Point ({POINT_X}, {POINT_Y}) is out of image bounds!")

    # normalizes to 0-255 for visualization
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # applys color map 
    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    # --- NEW: Visualize point on output image ---
    if 0 <= POINT_X < w_orig and 0 <= POINT_Y < h_orig:
        # Draw white circle
        cv2.circle(depth_color, (POINT_X, POINT_Y), 5, (255, 255, 255), -1)
        # Add text
        label = f"{depth[POINT_Y, POINT_X]:.2f}"
        cv2.putText(depth_color, label, (POINT_X + 10, POINT_Y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # saves
    cv2.imwrite("output_depth.jpg", depth_color)
    print("Result saved as 'output_depth.jpg'")

if __name__ == "__main__":
    run_inference()
