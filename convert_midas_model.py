import os
import numpy as np
import onnx
from PIL import Image
from hailo_sdk_client import ClientRunner

# config
model_name = "midas_v2_small"
onnx_path = "midas_v21_small_256.onnx"
calib_path = "calib_images"
chosen_hw_arch = "hailo8l"
MODEL_H, MODEL_W = 256, 256 

# extract node names from onnx
print(f"Untersuche {onnx_path}...")
model = onnx.load(onnx_path)
start_node = model.graph.input[0].name
end_node = model.graph.output[0].name
print(f"-> Gefundener Input-Name: '{start_node}'")
print(f"-> Gefundener Output-Name: '{end_node}'")

# init runner
runner = ClientRunner(hw_arch=chosen_hw_arch)

# load onnx
print("Lade ONNX Modell in Hailo...")
hn, npz = runner.translate_onnx_model(
    onnx_path,
    model_name,
    start_node_names=[start_node], 
    end_node_names=[end_node],   
    net_input_shapes={start_node: [1, 3, MODEL_H, MODEL_W]} 
)

# prepare calibration data
print("Lade Bilder für Kalibrierung...")
image_list = []
if not os.path.exists(calib_path):
    raise FileNotFoundError(f"Ordner {calib_path} fehlt!")

files = [f for f in os.listdir(calib_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# limit to 50 images
for filename in files[:50]:
    filepath = os.path.join(calib_path, filename)
    img = Image.open(filepath).convert('RGB').resize((MODEL_W, MODEL_H))
    img_data = np.array(img) 
    img_data = np.expand_dims(img_data, axis=0) 
    # normalize 0-1
    img_data = img_data.astype(np.float32) / 255.0 
    image_list.append(img_data)

full_dataset_array = np.concatenate(image_list, axis=0)

# optimization
# internal layer name differs from onnx name
calib_name = "midas_v2_small/input_layer1"
calib_data = { calib_name: full_dataset_array }

print("Starte Optimierung...")
runner.optimize(calib_data)

# compile
print("Kompiliere zu HEF...")
hef = runner.compile()

output_file = f"{model_name}_{chosen_hw_arch}.hef"
with open(output_file, "wb") as f:
    f.write(hef)

print(f"ERFOLG! Datei gespeichert als: {output_file}")
