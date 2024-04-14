#from ultralytics.utils import ASSETS
from ultralytics.models.yolo.detect import DetectionPredictor

args = dict(model='/Users/tianze/cs4243_lab/miniproject/jizhuo1105/runs/detect/train3/weights/best.pt',
            source="/Users/tianze/cs4243_lab/miniproject/jizhuo1105/datasets/test/knife")
predictor = DetectionPredictor(overrides=args)
predictor.predict_cli()

### python predict_script.py > results.txt
