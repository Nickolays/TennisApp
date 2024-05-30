import os
print(os.getcwd())

from src.scripts.detectors import CourtDetector, BallDetector, PlayerDetector
from src.scripts.RemoveBadFrames import RemoveBadFrames
from src.scripts import BallBounce, BallHit, Score, StepCount
from src.scripts import HomographyMatrix