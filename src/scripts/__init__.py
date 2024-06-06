import os
print(os.getcwd())

from src.scripts.detectors import CourtDetector, BallDetector, PlayerDetector
from src.scripts.RemoveBadFrames import RemoveBadFrames
from src.scripts import BallBounce, BallHit, Score, StepCount
from src.scripts.HomographyMatrix import HomographyMatrix

from src.scripts.RemoveShadow import ShadowRemover as RemoveShadow

from src.scripts.FillBallDetections import FillBallDetections
from src.scripts.ShotRecognition import ShotType