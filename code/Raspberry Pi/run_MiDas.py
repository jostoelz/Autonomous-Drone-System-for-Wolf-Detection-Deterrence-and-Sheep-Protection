import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import cv2
import os
import time

# --- CONFIGURATION ---
HEF_PATH = "midas_v2_small_hailo8_256_320.hef"
IMAGE_PATH = "Bilder/MiDas/Foto5.jpg" 

# Target pixel coordinates in the original image
TARGET_X_PIXEL = 112
TARGET_Y_PIXEL = 155

# Calibration factor for depth estimation
CALIBRATION_FACTOR = 1.0 

def run_midas_inference():
    # Load HEF
    try:
        hef = HEF(HEF_PATH)
    except Exception as e:
        print(f"HEF load error: {e}")
        return

    # Get model input requirements
    input_info = hef.get_input_vstream_infos()[0]
    model_h, model_w, _ = input_info.shape

    # Load original image
    img_orig = cv2.imread(IMAGE_PATH)
        
    orig_h, orig_w = img_orig.shape[:2]

    # PREPROCESSING: Resize / Stretch
    img_input = cv2.resize(img_orig, (model_w, model_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Prepare input buffer 
    input_data = np.expand_dims(img_input, axis=0).astype(np.uint8)
    input_data = np.ascontiguousarray(input_data)

    # Hailo Inference Pipeline
    with VDevice() as target:
        config_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, config_params)[0]
        network_params = network_group.create_params()
        
        input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        with network_group.activate(network_params):
            with InferVStreams(network_group, input_params, output_params) as pipeline:
                
                input_name = network_group.get_input_vstream_infos()[0].name
                output_name = network_group.get_output_vstream_infos()[0].name
                
                # Run inference and measure time
                t_start = time.perf_counter()
                results = pipeline.infer({input_name: input_data})
                t_end = time.perf_counter()
                
                # Calculate stats
                dt_ms = (t_end - t_start) * 1000
                fps = 1000 / dt_ms if dt_ms > 0 else 0
                
                # Extract raw output map (Height, Width)
                raw_out = results[output_name][0, :, :, 0]

                # --- COORDINATE MAPPING ---
                scale_x = model_w / orig_w
                scale_y = model_h / orig_h
                
                tx = int(TARGET_X_PIXEL * scale_x)
                ty = int(TARGET_Y_PIXEL * scale_y)

                # Clamp coordinates to ensure they remain within bounds
                tx = max(0, min(tx, model_w - 1))
                ty = max(0, min(ty, model_h - 1))

                # Get depth value at target pixel
                raw_val = raw_out[ty, tx]
                
                # Calculate inverse depth (avoid division by zero)
                dist = (CALIBRATION_FACTOR / raw_val) if raw_val > 1e-4 else 999.9

                print(f"Time:         {dt_ms:.2f} ms ({fps:.1f} FPS)")
                print(f"Raw Value:    {raw_val:.4f}")
                print(f"Distance:     {dist:.2f} m")

                # --- VISUALIZATION ---
                # Normalize output to 0-255 for visualization
                norm = cv2.normalize(raw_out, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                vis_map = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
                
                # Draw marker and distance text
                cv2.circle(vis_map, (tx, ty), 4, (255, 255, 255), 2)
                cv2.putText(vis_map, f"{dist:.2f}m", (tx+8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # Save result
                src_dir = os.path.dirname(IMAGE_PATH)
                if not src_dir: src_dir = "."
                src_base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
                save_path = os.path.join(src_dir, f"MiDas_{src_base}_result_256x320.jpg")
                
                cv2.imwrite(save_path, vis_map)
                print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    run_midas_inference()
