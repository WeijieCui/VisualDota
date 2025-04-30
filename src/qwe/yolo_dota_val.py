import os
from ultralytics import YOLO

if __name__ == '__main__':
    current = os.path.dirname(os.path.realpath(__file__))
    yama_file_name = 'DOTA_yolo.yaml'
    yaml = f"{current}/{yama_file_name}"

    # Load the model 记得替换这里的路径
    model = YOLO(r"C:\ding\好康的\qwe\runs\obb\train57\weights\best.pt")
    # Test the model
    results = model.val(data=yaml, imgsz=768, batch=2)