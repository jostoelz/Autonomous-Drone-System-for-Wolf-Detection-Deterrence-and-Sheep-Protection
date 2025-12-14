import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import cv2
import os
import time

# --- CONFIG ---
HEF_PATH = "scdepthv3.hef"
IMAGE_PATH = "Bilder/SCDepth/Foto5.jpg"

# Absolute coords in original image
TARGET_X_PIXEL = 112
TARGET_Y_PIXEL = 155

# Calibration (1.0 = Neutral)
# Formula: Dist = Factor / Sigmoid(Raw)
CALIBRATION_FACTOR = 1.0

def sigmoid(x):
    """Applies sigmoid function to map logits to (0, 1) range."""
    return 1 / (1 + np.exp(-x))

def resize_with_padding(image, target_size):
    """
    Resizes image to target_size maintaining aspect ratio.
    Returns: padded_image, (scale, pad_x, pad_y)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center offsets
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    return canvas, (scale, pad_x, pad_y)

def run_scdepth_measurement():
    print(f"--- START SC-DEPTH v3 ---")

    # Validate paths
    if not os.path.exists(HEF_PATH):
        print(f"ERR: HEF missing")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"ERR: Image missing")
        return

    # Load HEF
    try:
        hef = HEF(HEF_PATH)
    except Exception as e:
        print(f"HEF load error: {e}")
        return

    # Get model requirements
    input_info = hef.get_input_vstream_infos()[0]
    model_h, model_w, _ = input_info.shape
    print(f"-> Model req: {model_w}x{model_h}")

    # Load original image
    img_orig = cv2.imread(IMAGE_PATH)
    orig_h, orig_w = img_orig.shape[:2]
    print(f"-> Orig res:  {orig_w}x{orig_h}")

    # Pre-process: Resize with padding
    img_input, (scale, pad_x, pad_y) = resize_with_padding(img_orig, (model_w, model_h))

    input_data = np.expand_dims(img_input, axis=0).astype(np.uint8)
    input_data = np.ascontiguousarray(input_data)

    # Setup Hailo
    with VDevice() as target:
        config_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, config_params)[0]
        network_params = network_group.create_params()
        
        input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        print("Inference starting...")
        with network_group.activate(network_params):
            with InferVStreams(network_group, input_params, output_params) as pipeline:
                
                input_name = network_group.get_input_vstream_infos()[0].name
                output_name = network_group.get_output_vstream_infos()[0].name
                
                # Measure time
                t_start = time.perf_counter()
                results = pipeline.infer({input_name: input_data})
                t_end = time.perf_counter()
                
                # Stats
                dt_ms = (t_end - t_start) * 1000
                fps = 1000 / dt_ms if dt_ms > 0 else 0

                raw_out = results[output_name][0, :, :, 0]

                # Coordinate Mapping
                tx = int(TARGET_X_PIXEL * scale + pad_x)
                ty = int(TARGET_Y_PIXEL * scale + pad_y)

                # Clamp coords
                tx = max(0, min(tx, model_w - 1))
                ty = max(0, min(ty, model_h - 1))

                # Calc distance
                # SC-Depth specific: Raw -> Sigmoid -> Inverse
                raw_val = raw_out[ty, tx]
                disp = sigmoid(raw_val)
                
                dist = (CALIBRATION_FACTOR / disp) if disp > 1e-5 else 999.9

                print("\n" + "="*40)
                print(f"RESULT (SC-Depth Padded)")
                print(f"Time:         {dt_ms:.2f} ms ({fps:.1f} FPS)")
                print("-" * 40)
                print(f"Orig Pixel:   {TARGET_X_PIXEL}, {TARGET_Y_PIXEL}")
                print(f"Model Pixel:  {tx}, {ty}")
                print(f"Raw Logit:    {raw_val:.4f}")
                print(f"Sigmoid:      {disp:.4f}")
                print(f"Distance:     {dist:.2f} m")
                print("="*40)

                # Visualization
                # Normalize raw logits to 0-255 for display
                norm = cv2.normalize(raw_out, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                vis_map = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA) # Keeping Magma for SC-Depth style
                
                # Draw marker
                cv2.circle(vis_map, (tx, ty), 4, (255, 255, 255), 2)
                cv2.putText(vis_map, f"{dist:.2f}m", (tx+8, ty), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # Save Logic
                src_dir = os.path.dirname(IMAGE_PATH)
                src_base = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
                new_filename = f"SCDepth_{src_base}_result.jpg"
                out_path = os.path.join(src_dir, new_filename)
                
                cv2.imwrite(out_path, vis_map)
                print(f"Saved: {out_path}")

if __name__ == "__main__":
    run_scdepth_measurement()
