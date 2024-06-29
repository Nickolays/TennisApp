import os
print(os.getcwd())

from src.scripts.detectors import CourtDetector, BallDetector, PlayerDetector
from src.scripts.FillBallDetections import FillBallDetections

from src.scripts.BallBounce import BounceDetector, SecondBouncer
from src.scripts.HomographyMatrix import HomographyMatrix


from src.scripts.RemoveShadow import ShadowRemover as RemoveShadow
from src.scripts.RemoveBadFrames import RemoveBadFrames
from src.scripts.ShotRecognition import ShotType
# from src.scripts import BallHit, Score, StepCount  