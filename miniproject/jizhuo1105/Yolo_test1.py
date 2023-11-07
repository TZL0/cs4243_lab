from ultralytics import YOLO

# YOLOv8n	640	37.3	80.4	0.99	3.2	8.7
# YOLOv8s	640	44.9	128.4	1.20	11.2	28.6
# YOLOv8m	640	50.2	234.7	1.83	25.9	78.9
# YOLOv8l	640	52.9	375.2	2.39	43.7	165.2
# YOLOv8x	640	53.9	479.1	3.53	68.2	257.8

# randomly take 100 images from images folder
# get all file names in images folder
import os
import random
import shutil


def get_all_file_names(path):
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)
    return files


def get_random_files(path, num):
    files = get_all_file_names(path)
    random.shuffle(files)
    return files[:num]


def random_select_100_files(path="/Users/tianze/cs4243_lab/miniproject/jizhuo1105/datasets/"):
    image_train_path = os.path.join(path, "images/train")
    image_val_path = os.path.join(path, "images/val")

    label_train_path = os.path.join(path, "labels/train")
    label_val_path = os.path.join(path, "labels/val")

    files = get_random_files(image_train_path, 100)

    for file in files:
        shutil.move(os.path.join(image_train_path, file), image_val_path)
        # get the file name without extension
        annotation_txt = os.path.splitext(file)[0] + '.txt'
        shutil.move(os.path.join(label_train_path, annotation_txt), label_val_path)


def return_selected_files(path="/Users/tianze/cs4243_lab/miniproject/jizhuo1105/datasets"):
    image_train_path = os.path.join(path, "images/train")
    image_val_path = os.path.join(path, "images/val")

    label_train_path = os.path.join(path, "labels/train")
    label_val_path = os.path.join(path, "labels/val")

    files = get_all_file_names(image_val_path)

    for file in files:
        shutil.move(os.path.join(image_val_path, file), image_train_path)
        # get the file name without extension
        annotation_txt = os.path.splitext(file)[0] + '.txt'
        shutil.move(os.path.join(label_val_path, annotation_txt), label_train_path)



random_select_100_files()

# # Load a model
model = YOLO("yolov8s.pt")  # build a new model from scratch
model2 = YOLO("yolov8l.pt")  # build a new model from scratch

# Use the model
results = model.train(data="/Users/tianze/cs4243_lab/miniproject/jizhuo1105/douxing.yaml", epochs=16)  # train the model
results2 = model2.train(data="/Users/tianze/cs4243_lab/miniproject/jizhuo1105/douxing.yaml", epochs=16)  # train the model

return_selected_files()