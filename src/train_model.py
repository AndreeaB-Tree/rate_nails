from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO('yolov8s-seg.pt')

# Path to dataset YAML file
dataset_path = "data/nails_segmentation/dataset.yaml"

# Train the model
model.train(data=dataset_path, epochs=50, task='segment')

# Validate and export the model
metrics = model.val()
model.export("models/segmentation_model.pt")
