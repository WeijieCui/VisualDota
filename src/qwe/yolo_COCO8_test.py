from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)