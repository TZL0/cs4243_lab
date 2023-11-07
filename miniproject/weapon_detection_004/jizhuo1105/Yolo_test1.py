from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # build a new model from scratch
model2 = YOLO("yolov8l.pt")  # load a model from a path

# Use the model
results = model.train(data="douxing.yaml", epochs=16)  # train the model