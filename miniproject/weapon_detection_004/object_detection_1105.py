import torch
from ultralytics import YOLO

# Load a pre-trained YOLOv5 model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Set up training hyperparameters (this is just a starting point)
hyp = {'lr0': 0.01,  # initial learning rate
       'lrf': 0.1,   # final learning rate
       'momentum': 0.937,
       'weight_decay': 0.0005,
       'warmup_epochs': 3.0,
       'warmup_momentum': 0.8,
       'warmup_bias_lr': 0.1,
       'box': 0.05,
       'cls': 0.5,
       'cls_pw': 1.0,
       'obj': 1.0,
       'obj_pw': 1.0,
       'iou_t': 0.20,
       'anchor_t': 4.0,
       'fl_gamma': 0.0,
       'hsv_h': 0.015,
       'hsv_s': 0.7,
       'hsv_v': 0.4,
       'degrees': 0.0,
       'translate': 0.1,
       'scale': 0.5,
       'shear': 0.0,
       'perspective': 0.0,
       'flipud': 0.0,
       'fliplr': 0.5,
       'mosaic': 1.0,
       'mixup': 0.0}

# Set up training configuration
cfg = {'epochs': 100,            # number of epochs
       'batch_size': 16,         # batch size
       'accumulate': 1,          # accumulate loss before backward
       'img_size': 640,          # input image size
       'multi_scale': False,     # use multi-scale image sizing during training
       'rect': False,            # use rectangular training
       'resume': False,          # resume training from last.pt
       'nosave': False,          # only save final checkpoint
       'noval': False,           # only validate final epoch
       'noautoanchor': False,    # disable autoanchor check
       'evolve': None,           # evolve hyperparameters
       'bucket': '',             # gsutil bucket
       'cache_images': False,    # cache images for faster training
       'image_weights': False,   # use image weights
       'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # training device
       'multi_gpu': False,       # use multiple GPUs
       'single_cls': False,      # train as single-class dataset
       'adam': False,            # use torch.optim.Adam() optimizer
       'sync_bn': False,         # use SyncBatchNorm, only available in DDP mode
       'workers': 8,             # max dataloader workers
       'project': 'runs/train',  # save to project/name
       'name': 'exp',            # save to project/name
       'exist_ok': False,        # existing project/name ok, do not increment
       'quad': False,            # use QuadBench data loader
       'linear_lr': False,       # linear LR
       'label_smoothing': 0.0,   # Label smoothing epsilon
       'bbox_interval': -1,      # bbox validation interval
       'save_period': -1,        # checkpoint save period
       'artifact_alias': 'latest'  # version name (default is 'latest')
       }

# Set up data configuration
data = {'train': 'train/',  # path to train images
        'val': 'val/',      # path to validation images
        'nc': 2,                          # number of classes
        'names': ['gun', 'knife']  # class names
       }


# Load a model
model = YOLO("yolov8s.pt")  # build a new model from scratch
model2 = YOLO("yolov8l.pt")  # build a new model from scratch


# Use the model
results = model.train(data='data.yaml', epochs=16)  # train the model
results2 = model2.train(data='data.yaml', epochs=16)  # train the model
