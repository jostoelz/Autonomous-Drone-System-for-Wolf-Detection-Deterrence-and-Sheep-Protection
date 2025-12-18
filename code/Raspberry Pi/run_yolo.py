import numpy as np
from PIL import Image, ImageDraw, ImageFont
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

# --- CONFIGURATION ---
HEF_FILE = "yolov8s.hef"
IMAGE_FILE = "Bilder/MiDas/Wolf2.jpg"
OUTPUT_FILE = "ergebnis.jpg"

def letterbox_image(image, target_width, target_height):
    """
    Resizes image with unchanged aspect ratio using padding.
    This is necessary because the model expects a specific input size (e.g., 640x640).
    """
    img_w, img_h = image.size
    # Calculate the scaling factor to fit the image into the target dimensions
    scale = min(target_width / img_w, target_height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Resize using high-quality Lanczos filtering
    image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Create a new gray canvas (114 is the standard padding color for YOLO)
    new_image = Image.new("RGB", (target_width, target_height), (114, 114, 114))
    
    # Center the resized image on the canvas
    paste_x = (target_width - new_w) // 2
    paste_y = (target_height - new_h) // 2
    new_image.paste(image, (paste_x, paste_y))

    # Return the image as a numpy array and the offsets/scale for coordinate reconstruction later
    return np.array(new_image), (scale, paste_x, paste_y)

def infer():
    # Load the compiled Hailo model (HEF)
    hef = HEF(HEF_FILE)

    # Context manager handles the connection to the Hailo-8/8L hardware
    with VDevice() as target:
        # Standard configuration for PCIe-based Hailo devices
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        # Define stream formats: images in as UINT8, detections out as FLOAT32
        input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        # Get the input dimensions the model expects
        input_vstream_info = hef.get_input_vstream_infos()[0]
        model_h = input_vstream_info.shape[0]
        model_w = input_vstream_info.shape[1]

        # 1. Load and Preprocess
        original_image = Image.open(IMAGE_FILE)
        processed_image, (scale, pad_x, pad_y) = letterbox_image(original_image, model_w, model_h)
        
        # Add batch dimension (1, H, W, C)
        input_data = np.expand_dims(processed_image, axis=0)

        # Start the inference pipeline
        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                infer_results = infer_pipeline.infer(input_data)

                # Setup PIL drawing context
                draw = ImageDraw.Draw(original_image)
                
                # Load a font for labels, fall back to default if not found
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 20)
                except:
                    font = ImageFont.load_default()

                # Get the detection results (usually stored in the first output stream)
                key = list(infer_results.keys())[0]
                batch_result = infer_results[key]
                detections_per_class = batch_result[0]
                
                total_detections = 0

                # Iterate through all detected objects across all classes
                for class_id, class_detections in enumerate(detections_per_class):
                    num_in_class = len(class_detections)
                    
                    if num_in_class > 0:
                        total_detections += num_in_class
                        
                        for i in range(num_in_class):
                            detection = class_detections[i]
                            ymin, xmin, ymax, xmax, score = detection
                            
                            # --- 1. De-Normalize (Convert percentages back to model pixel values) ---
                            if xmax <= 1.5 and ymax <= 1.5:
                                ymin = ymin * model_h
                                xmin = xmin * model_w
                                ymax = ymax * model_h
                                xmax = xmax * model_w
                            
                            # --- 2. Post-processing (Map model pixels back to original image scale) ---
                            # Subtract the padding and then divide by the scale factor
                            x1 = (xmin - pad_x) / scale
                            y1 = (ymin - pad_y) / scale
                            x2 = (xmax - pad_x) / scale
                            y2 = (ymax - pad_y) / scale

                            # Clip coordinates to ensure they stay within the image boundaries
                            x1 = max(0, min(original_image.width, x1))
                            y1 = max(0, min(original_image.height, y1))
                            x2 = max(0, min(original_image.width, x2))
                            y2 = max(0, min(original_image.height, y2))

                            # --- 3. Visualization ---
                            # Draw the bounding box
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                            
                            # Label formatting
                            label_text = f"Class {class_id}: {score:.2f}"
                            
                            # Draw a background rectangle for the text to make it readable
                            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
                            draw.rectangle(text_bbox, fill="red")
                            
                            # Draw the text label
                            draw.text((x1, y1), label_text, fill="white", font=font)

                # Only save if we actually found something to show
                if total_detections > 0:
                    original_image.save(OUTPUT_FILE)
                    print(f"Done! Saved result to: {OUTPUT_FILE}")

if __name__ == "__main__":
    infer()
