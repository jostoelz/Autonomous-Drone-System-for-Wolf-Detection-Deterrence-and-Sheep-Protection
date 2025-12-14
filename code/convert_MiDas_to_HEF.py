import os
import numpy as np
import onnx
from PIL import Image
from hailo_sdk_client import ClientRunner

# --- Configuration ---
model_name = "midas_v2_small"
onnx_path = "midas_v2_small_256x320.onnx"
calib_path = "calib_images"
hw_arch = "hailo8"

# Input resolution (HxW). Must match ONNX export exactly.
MODEL_H, MODEL_W = 256, 320

# Parse ONNX to get original I/O names
model = onnx.load(onnx_path)
start_node = model.graph.input[0].name
end_node = model.graph.output[0].name

# Init Hailo ClientRunner
runner = ClientRunner(hw_arch=hw_arch)

# Translate ONNX to Hailo internal format (HAR)
hn, npz = runner.translate_onnx_model(
    onnx_path,
    model_name,
    start_node_names=[start_node],
    end_node_names=[end_node],
    net_input_shapes={start_node: [1, 3, MODEL_H, MODEL_W]}
)

# Prepare calibration dataset
image_list = []

# Check if calibration images exist, else fallback to random noise
if not os.path.exists(calib_path) or not os.listdir(calib_path):
    for _ in range(20):
        # Generate random noise (uint8)
        img_data = np.random.randint(0, 255, (MODEL_H, MODEL_W, 3), dtype=np.uint8)
        img_data = np.expand_dims(img_data, axis=0)    # Add batch dim
        img_data = img_data.astype(np.float32) / 255.0 # Normalize 0..1
        image_list.append(img_data)
else:
    # Load real images
    files = [f for f in os.listdir(calib_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Cap at 50 images for speed
    for filename in files[:50]:
        filepath = os.path.join(calib_path, filename)
        # Resize to specific model input (256x320)
        img = Image.open(filepath).convert('RGB').resize((MODEL_W, MODEL_H), Image.Resampling.LANCZOS)
        img_data = np.array(img)
        img_data = np.expand_dims(img_data, axis=0)    # Add batch dim
        img_data = img_data.astype(np.float32) / 255.0 # Normalize 0..1
        image_list.append(img_data)

calib_dataset = np.concatenate(image_list, axis=0)

# Optimization
hailo_input_name = "midas_v2_small/input_layer1"
calib_data = {hailo_input_name: calib_dataset}

runner.optimize(calib_data)

# Compile to HEF
hef = runner.compile()

# Save output
output_file = f"{model_name}_{hw_arch}.hef"
with open(output_file, "wb") as f:
    f.write(hef)

print(f"SUCCESS! Saved to: {output_file}")
