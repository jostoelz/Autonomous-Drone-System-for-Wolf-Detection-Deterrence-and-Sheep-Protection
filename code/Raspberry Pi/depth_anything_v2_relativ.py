import cv2
import numpy as np
import onnxruntime as ort
import time
import os

MODEL_PATH = "depth_anything_v2_vitb_dynamic.onnx" 
IMAGE_PATH = "IMG_3796.JPEG"
INPUT_SIZE = (518, 518)

# Coordinates 
POINT_X = 780
POINT_Y = 1000

# The output values are abstract, so keep this at 1.0
SCALE_FACTOR = 1.0 

def letterbox_image(image, size):
    """Resizes image preserving aspect ratio and padding with gray."""
    ih, iw = image.shape[:2]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh))
    
    # Create a gray image 
    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    
    # Center the image
    dy = (h - nh) // 2
    dx = (w - nw) // 2
    new_image[dy:dy+nh, dx:dx+nw, :] = image_resized
    return new_image, (scale, dx, dy)

def run_inference():
    print(f"Loading model: {MODEL_PATH}...")
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_name = session.get_inputs()[0].name
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error loading image: {IMAGE_PATH}")
        return

    h_orig, w_orig = image.shape[:2]

    # --- Preprocessing ---
    image_input_viz, (scale, pad_x, pad_y) = letterbox_image(image, INPUT_SIZE)
    image_rgb = cv2.cvtColor(image_input_viz, cv2.COLOR_BGR2RGB)
    
    # Standard ImageNet Normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_rgb / 255.0 - mean) / std
    
    image_input = image_norm.transpose(2, 0, 1).astype(np.float32)
    image_input = np.expand_dims(image_input, axis=0)

    # --- Inference ---
    print("Starting inference...")
    start_time = time.time()
    outputs = session.run(None, {input_name: image_input})
    print(f"Inference Time: {time.time() - start_time:.4f} seconds")

    raw_depth = outputs[0].squeeze()

    nh = int(h_orig * scale)
    nw = int(w_orig * scale)
    
    # Extract only the valid area
    valid_depth = raw_depth[pad_y:pad_y+nh, pad_x:pad_x+nw]
    
    # Resize to original dimensions
    depth_final = cv2.resize(valid_depth, (w_orig, h_orig))

    # Normalizes the entire image to a range of 0.0 to 1.0.
    # 0.0 = Furthest point 
    # 1.0 = Closest point 
    d_min = depth_final.min()
    d_max = depth_final.max()
    
    if d_max - d_min > 0:
        depth_relative = (depth_final - d_min) / (d_max - d_min)
    else:
        depth_relative = np.zeros_like(depth_final)

    # Check specific point
    if 0 <= POINT_X < w_orig and 0 <= POINT_Y < h_orig:
        # Retrieve the value between 0.0 and 1.0
        val = depth_relative[POINT_Y, POINT_X]
        
        print(f"Coordinates: X={POINT_X}, Y={POINT_Y}")
        print(f"Raw Value: {depth_final[POINT_Y, POINT_X]:.4f}")
        print(f"Relative Score: {val:.4f} (0=Far/Ground, 1=Near/Obstacle)")
        
    else:
        print("Point out of bounds.")
    
    # Visualization (Heatmap)
    depth_uint8 = (depth_relative * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    
    # Draw point on output
    if 0 <= POINT_X < w_orig and 0 <= POINT_Y < h_orig:
        cv2.circle(depth_color, (POINT_X, POINT_Y), 5, (255, 255, 255), -1)
        # Writes the relative score into the image
        label = f"{depth_relative[POINT_Y, POINT_X]:.2f}" 
        cv2.putText(depth_color, label, (POINT_X + 10, POINT_Y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Extract filename from path
    base_name = os.path.basename(IMAGE_PATH)
    # Remove extension 
    file_name_no_ext = os.path.splitext(base_name)[0]
    # Create new name
    output_filename = f"{file_name_no_ext}_relativ.jpg"

    cv2.imwrite(output_filename, depth_color)
    print(f"Saved {output_filename}")

if __name__ == "__main__":
    run_inference()
