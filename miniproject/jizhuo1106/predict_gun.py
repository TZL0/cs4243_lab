#from ultralytics.utils import ASSETS
from ultralytics.models.yolo.detect import DetectionPredictor

args = dict(model='D:\\mini_project\\datasets\\runs\\detect\\train6\\weights\\best.pt', 
            source="D:\\mini_project\\datasets\\test\\gun")
predictor = DetectionPredictor(overrides=args)
predictor.predict_cli()

### python predict_script.py > results.txt
