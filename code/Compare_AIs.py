import os
import time
import base64
import re
import cv2
import pandas as pd
from openai import OpenAI
from ultralytics import YOLO
import mimetypes

# --- CONFIGURATION ---
# Initialize the client.
client = OpenAI(api_key="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") 

MODEL_PATH = "best.pt"
TEST_DIR = "test"  # This is the root folder containing 'images' and 'labels' subfolders
OUTPUT_FOLDER = "benchmark_final"

# Mapping class IDs to human-readable names based on the training data
CLASS_MAP = {0: "dog", 1: "sheep", 2: "wolf"}

# Load the custom YOLO model once at the start
yolo_model = YOLO(MODEL_PATH)

# Ensure the output directory exists so we don't crash later
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- UTILITY FUNCTIONS ---

def yolo_to_corners(yolo_box):
    """
    Converts YOLO format [class, center_x, center_y, width, height] 
    to standard bounding box format [xmin, ymin, xmax, ymax].
    """
    cls, xc, yc, w, h = yolo_box
    # Calculate corners based on the center point
    xmin = xc - w / 2
    ymin = yc - h / 2
    xmax = xc + w / 2
    ymax = yc + h / 2
    return int(cls), [xmin, ymin, xmax, ymax]

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU).
    This is the standard metric to see how much two boxes overlap.
    Returns a value between 0 (no overlap) and 1 (perfect match).
    """
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both bounding boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute IoU, adding a tiny epsilon (1e-6) to avoid division by zero errors
    return inter / float(area1 + area2 - inter + 1e-6)

def get_gpt_prediction(image_path):
    """
    Sends the image to GPT-4o to get bounding box predictions.
    Includes error handling and specific prompting strategies.
    """
    # 1. Determine the correct MIME type (e.g., image/jpeg or image/png)
    # The API is picky about this, so we guess based on the file extension.
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg" # Fallback if we can't guess

    # Encode the image to base64 string
    with open(image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode("utf-8")
    
    # 2. Construct the Prompt (Experimental "Chain of Thought")
    # We ask GPT to first describe the scene, then list coordinates. 
    # Forcing it to "think" before outputting numbers usually improves accuracy.
    prompt = (
        "Identify every wolf, dog, and sheep in this photo. "
        "First, briefly state what you see. "
        "Then, provide the coordinates in this strict format: "
        "label: [xmin, ymin, xmax, ymax] | ... "
        "Use normalized 0.0 to 1.0 coordinates. If truly nothing is found, say 'None'."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{base64_img}",
                    "detail": "high" # High detail mode consumes more tokens but sees small objects better
                }}
            ]}],
            max_tokens=500,
            temperature=0.2 # Low temperature keeps the formatting consistent/deterministic
        )
        text = response.choices[0].message.content.lower()
        
        # Debug print: useful to see if GPT is chatting too much or hallucinating
        print(f"DEBUG GPT full response for {os.path.basename(image_path)}: {text}")
        
        dets = []
        # 3. Parse the response
        # Since LLMs can be chatty, we split by the pipe character '|' and use Regex 
        # to extract the label and the four coordinates.
        items = text.split("|")
        for item in items:
            label_match = re.search(r'(dog|sheep|wolf)', item)
            # Find all floating point numbers (e.g., 0.5, 1.0, 0.23)
            coords = re.findall(r"0\.\d+|1\.0|\d\.\d+", item)
            
            # We only keep it if we found a valid label AND exactly 4 coordinates
            if label_match and len(coords) == 4:
                dets.append({
                    "label": label_match.group(0), 
                    "box": [float(c) for c in coords]
                })
        return dets

    except Exception as e:
        print(f"GPT API Error: {e}")
        return []

# --- MAIN ENGINE ---

image_dir = os.path.join(TEST_DIR, "images")
label_dir = os.path.join(TEST_DIR, "labels")
results = []

print("Starting deep performance comparison with image export...")

# Iterate through every image in the test folder
for img_name in os.listdir(image_dir):
    # Skip non-image files
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    img_path = os.path.join(image_dir, img_name)
    # Construct the corresponding label path (assumes same filename, but .txt extension)
    lbl_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
    
    # If there is no ground truth label, we can't benchmark, so skip it
    if not os.path.exists(lbl_path):
        continue

    # Load image using OpenCV for visualization
    img_cv = cv2.imread(img_path)
    h, w, _ = img_cv.shape

    # ---------------------------------------------------------
    # 1. PROCESS GROUND TRUTH (The "Correct" Answers)
    # ---------------------------------------------------------
    gt_objects = []
    with open(lbl_path, "r") as f:
        for line in f:
            data = [float(x) for x in line.split()]
            cls, box = yolo_to_corners(data)
            gt_objects.append({"label": CLASS_MAP[cls], "box": box})
            
            # DRAW GROUND TRUTH: Red boxes
            # We convert normalized coordinates (0-1) back to pixel coordinates here
            cv2.rectangle(img_cv, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0, 0, 255), 3)
            cv2.putText(img_cv, f"GT:{CLASS_MAP[cls]}", (int(box[0]*w), int(box[1]*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ---------------------------------------------------------
    # 2. GET YOLO PREDICTIONS
    # ---------------------------------------------------------
    y_start = time.time()
    y_res = yolo_model(img_path, verbose=False)[0]
    y_time = (time.time() - y_start) * 1000 # Convert to milliseconds
    
    y_preds = []
    for b in y_res.boxes:
        box = b.xyxyn[0].cpu().numpy().tolist() # Get normalized coordinates
        label = CLASS_MAP[int(b.cls[0])]
        y_preds.append({"label": label, "box": box})
        
        # DRAW YOLO: Green boxes
        cv2.rectangle(img_cv, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0, 255, 0), 2)
        cv2.putText(img_cv, f"YOLO:{label}", (int(box[0]*w), int(box[3]*h)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------------------------------------------------------
    # 3. GET GPT PREDICTIONS
    # ---------------------------------------------------------
    g_start = time.time()
    g_preds = get_gpt_prediction(img_path)
    g_time = (time.time() - g_start) * 1000 # Convert to milliseconds
    
    for gp in g_preds:
        box = gp["box"]
        # DRAW GPT: Blue boxes
        cv2.rectangle(img_cv, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (255, 0, 0), 2)
        # Position text slightly differently to avoid overlap with YOLO text
        cv2.putText(img_cv, f"GPT:{gp['label']}", (int(box[2]*w)-80, int(box[1]*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 4. Save the annotated comparison image
    output_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(output_path, img_cv)

    # ---------------------------------------------------------
    # 5. CALCULATE METRICS (IoU)
    # ---------------------------------------------------------
    # We loop through every Ground Truth object and find the best matching prediction
    for gt in gt_objects:
        # Check YOLO matches
        y_ious = [calculate_iou(gt["box"], p["box"]) for p in y_preds]
        best_y_iou = max(y_ious) if y_ious else 0
        
        # Check GPT matches
        g_ious = [calculate_iou(gt["box"], p["box"]) for p in g_preds]
        best_g_iou = max(g_ious) if g_ious else 0

        # Log the data for this specific object
        results.append({
            "image": img_name,
            "class": gt["label"],
            "yolo_iou": best_y_iou,
            "gpt_iou": best_g_iou,
            "yolo_latency_ms": y_time,
            "gpt_latency_ms": g_time,
            # We consider it "found" if the Intersection over Union is greater than 45%
            "yolo_found": 1 if best_y_iou > 0.45 else 0,
            "gpt_found": 1 if best_g_iou > 0.45 else 0
        })

# Final Step: Save all statistical data to CSV for analysis
df = pd.DataFrame(results)
df.to_csv("detailed_comparison.csv", index=False)
print(f"Done! Annotated images saved to '{OUTPUT_FOLDER}' and stats saved to CSV.")
