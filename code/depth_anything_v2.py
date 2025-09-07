# preparation
!git clone https://github.com/DepthAnything/Depth-Anything-V2
%cd Depth-Anything-V2/metric_depth
!pip install -r requirements.txt

# downloads the checkpoints and puts them under the checkpoints directory.
!mkdir -p checkpoints
!wget -O checkpoints/depth_anything_v2_metric_vkitti_vits.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true"

import cv2
import torch
from google.colab import drive
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt

# paths
drive.mount('/content/drive')
input_image_folder = '/content/drive/MyDrive/ColabDatasets/Training/Neue_Bilder/frame_17946.png'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits' # or 'vitl', 'vitb'
dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 80 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model.eval()

raw_img = cv2.imread(input_image_folder)
depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
# shows depth map as an image
plt.imshow(depth, cmap="plasma")
plt.colorbar(label="Tiefe (m)")
plt.show()
