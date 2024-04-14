from ultralytics import YOLO

# model = YOLO('yolov8n-cls.pt')

# model.train(data='/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001',
#             epochs=8,
#             imgsz=256)

model = YOLO('/Users/tianze/cs4243_lab/miniproject/jizhuo1114/runs/classify/train4/weights/best.pt')

# run yolo model on test dataset and draw
model.predict('/Users/tianze/cs4243_lab/miniproject/drive-download-20230925T130828Z-001/test/')
