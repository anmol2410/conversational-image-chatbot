from ultralytics import YOLO

# Load YOLO model once (nano version for speed)
model = YOLO('yolov8n.pt')

def detect_objects(image_path):
    """
    Detect objects in an image and return list of detected labels.
    Example: ['dog', 'person', 'car']
    """
    results = model(image_path)
    labels = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy())   # class ID
            labels.append(model.names[cls])    # convert to class name
    return labels
