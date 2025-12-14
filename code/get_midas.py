import torch

# Load MiDaS v2.1 Small from Intel Hub
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
model.eval()
model.to("cpu")

# Prepare dummy input
height, width = 256, 320
dummy_input = torch.randn(1, 3, height, width)

# Export to ONNX
output_file = "midas_v2_small_256x320.onnx"

torch.onnx.export(
    model,
    dummy_input,
    output_file,
    opset_version=11,        # Recommended for Hailo
    input_names=['input'],   # Fixed input name
    output_names=['output'], # Fixed output name
    do_constant_folding=True
)

print("DONE! Model ready for Hailo compilation.")
