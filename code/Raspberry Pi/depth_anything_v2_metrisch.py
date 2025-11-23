import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "depth_anything_v2_vits_518x518.onnx" 
IMAGE_PATH = "IMG_3796.JPEG"
INPUT_SIZE = (518, 518) # Model input size

# Coordinates (on the original image!)
POINT_X = 780 
POINT_Y = 1000 

SCALE_FACTOR = 1.0 # Keep at 1.0 initially, then adjust based on a real measurement.

def letterbox_image(image, size):
    """Resizes image preserving aspect ratio and padding with gray."""
    ih, iw = image.shape[:2]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_resized = cv2.resize(image, (nw, nh))
    
    # Create a gray image (128 is neutral for ImageNet normalization)
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

    # --- 1. Correct Resizing (Letterbox) ---
    # Prevents distortion that makes objects look thinner (and thus further away)
    image_input_viz, (scale, pad_x, pad_y) = letterbox_image(image, INPUT_SIZE)
    
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_input_viz, cv2.COLOR_BGR2RGB)
    
    # Normalize (Standard ImageNet values)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = (image_rgb / 255.0 - mean) / std
    
    # HWC to CHW
    image_input = image_norm.transpose(2, 0, 1).astype(np.float32)
    # Add batch dimension
    image_input = np.expand_dims(image_input, axis=0)

    # Inference
    print("Starting inference...")
    start_time = time.time()
    outputs = session.run(None, {input_name: image_input})
    print(f"Inference Time: {time.time() - start_time:.4f} seconds")

    raw_depth = outputs[0][0] # Shape: (518, 518)
    
    nh = int(h_orig * scale)
    nw = int(w_orig * scale)
    
    # Extract only the area containing the valid image
    valid_depth = raw_depth[pad_y:pad_y+nh, pad_x:pad_x+nw]
    
    # Resize to original dimensions
    depth_final = cv2.resize(valid_depth, (w_orig, h_orig))

    # --- 3. Apply Calibration ---
    depth_final = depth_final * SCALE_FACTOR

    # Check specific point
    if 0 <= POINT_X < w_orig and 0 <= POINT_Y < h_orig:
        depth_value = depth_final[POINT_Y, POINT_X]
        print(f"Coordinates: X={POINT_X}, Y={POINT_Y}")
        print(f"Measured Depth: {depth_value:.2f} (Unit depends on calibration)")
    else:
        print("Point out of bounds.")
    
    # Visualization
    depth_min = depth_final.min()
    depth_max = depth_final.max()
    # Normalize to 0-255 for display
    depth_normalized = (depth_final - depth_min) / (depth_max - depth_min) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    
    # Draw point on output
    if 0 <= POINT_X < w_orig and 0 <= POINT_Y < h_orig:
        cv2.circle(depth_color, (POINT_X, POINT_Y), 5, (255, 255, 255), -1)
        cv2.putText(depth_color, f"{depth_final[POINT_Y, POINT_X]:.2f}", 
                    (POINT_X + 10, POINT_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite("output_depth_corrected.jpg", depth_color)
    print("Saved output_depth_corrected.jpg")

if __name__ == "__main__":
    run_inference()
