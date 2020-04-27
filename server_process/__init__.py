from .sim_classifier import classifier as ActClassifier
from .estimator import PoseEstimator
from .webcam_detector import WebCamDetectionLoader as WebcamDetector
from .mainProcess import mainProcess as MainProcess
__all__ = [
    'PoseClassifier', 'PoseEstimator','WebcamDetector','MainProcess'
]