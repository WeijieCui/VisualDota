import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from ultralytics import YOLO
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
if __name__ == '__main__':
    print("CUDA Available:", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    current_folder = os.path.dirname(os.path.abspath(__file__))
    yaml_file_name = 'DOTA_yolo.yaml'
    yaml = f"{current_folder}/{yaml_file_name}"
    # Load a model
    model = YOLO("yolo11n-obb.yaml").load("yolo11n.pt")
    # Train the model
    results = model.train(data=yaml, epochs=12, imgsz=768, batch=2)
