"""
    The first version of this progect, frame consist from torch's arrays. Type of frame is List. And then we use List everyvere


    BIG TODO: 
        - To come up with new NN, for removing frames without game time, for instance: below the net, after out, and so on
        - Shadow remover

"""


import os
from typing import List

import pandas as pd
import cv2
import pickle   # for artifacts
import torch

from src.scripts import CourtDetector, PlayerDetector, BallDetector
from src.scripts import RemoveBadFrames, RemoveShadow
from src.scripts.utils import read_video, save_video
# from models import 


# 
MAIN_PATH = os.getcwd()
SHIFT_FRAMES_2_PREDICTIONS = 25   # Ball
# VIDEO_SIZE = (1080, 1920)
# FPS = 30 

# 
path = r'data/vizualize/tennis_1_new.mp4'  # r'data/vizualize/input_video.mp4'    # 
is_writen = True    # Artefacts of trackers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# 
if os.path.exists(path):
    # Get Images
    frames = read_video(path)
    print(f"Frames length is:   {len(frames)}\n")
else:
    os.chdir("/home/suetin/Projects/VSCode/TennisApp/MainApp/TennisApp/")
    frames = read_video(path)
    print(f"Frames length is:   {len(frames)}\n")


# Load Models
court_model_path = './models/keypoints_model.pth'  # './models/court_model.pt'
ball_model_path = './models/ball_model.pt'
player_model_path = './models/player_model.pt'

print("Test paths with models")
print(os.path.exists(court_model_path))
print(os.path.exists(ball_model_path))
print(os.path.exists(player_model_path))
print()

# 
#       Clear Game
#   Removing frames without game time
remover_path = ''
remover = RemoveBadFrames(remover_path)
frames = remover(frames)

# 
#       Machine Learning Block
#   Init our fitted and prepared model
FieldsKeypointsDetection = CourtDetector(court_model_path)
players_detector = PlayerDetector(player_model_path)
ball_detectior = BallDetector(ball_model_path)
#   Apply Machine Learning Block
# Get Model predict from input video
player_predictions = players_detector.predict(frames,
                                               read_from_stub=is_writen,
                                               stub_path='data/temporary/player_detections.pkl')  # players_detector(video) 
# Get predicts of balls
ball_predictions = ball_detectior.predict(frames, 
                                          read_from_stub=is_writen,
                                          stub_path='data/temporary/ball_detections.pkl') 
# Get court keypoints 
court_keypoints_predictions = FieldsKeypointsDetection.predict(frames,
                                                               read_from_stub=is_writen,
                                                               stub_path='data/temporary/court_detections.pkl') 

output_video_1 = FieldsKeypointsDetection.draw_keypoints(frames, court_keypoints_predictions)
output_video_1 = ball_detectior.draw_bboxes(video_frames=output_video_1, ball_detections=ball_predictions)
output_video_path = "./results/output_1.mp4"
save_video(output_video_1, output_video_path)


# 1. Use other model, for Shot Recognition and Predicting missing points
# Predict missing points in ball detections
from src.scripts import FillBallDetections, ShotType

# Concat ball detections and Type of shoе predict (classification)
# results


shot_recognitor = ShotType()  # Use predicted Players coordinates of body keypoints
types_of_shot = shot_recognitor(player_predictions)

assert len(ball_predictions) == len(frames)

import numpy as np
array_ball_predictions = np.array(ball_predictions)    # NEW FOR TRY OTHER INTERPOLATION

ball_filler = FillBallDetections()
ball_predictions = ball_filler.preprocess(ball_predictions, window=5)   # NEW
# ball_predictions = ball_filler.choose_main_ball(ball_predictions)
# assert len(ball_predictions) == len(frames)
# ball_predictions = ball_filler.interpolate_ball_position(ball_predictions) 

assert len(ball_predictions) == len(frames)

# Choose players !!! ERROR HERE !!!
player_predictions = players_detector.choose_and_filter_players(court_keypoints_predictions, player_predictions)


output_video_2 = frames.copy()
output_video_2 = FieldsKeypointsDetection.draw_keypoints(output_video_2, court_keypoints_predictions)
output_video_2 = ball_detectior.draw_bboxes(video_frames=output_video_2, ball_detections=ball_predictions)
output_video_2 = players_detector.draw_bboxes(output_video_2, player_predictions)

# Draw frame number on top left corner
for i, frame in enumerate(output_video_2):
    output_video_2[i] = cv2.putText(output_video_2[i], f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

output_video_path = "./results/output_2.mp4"
save_video(output_video_2, output_video_path)
# 
#       Classical Programming and Classical CV
#    Обработаем выход из модели. 
# https://github.com/ArtLabss/tennis-tracking/blob/main/court_detector.py

# 1. Update homography condition
from src.scripts import HomographyMatrix
import numpy as np
from cv2 import getPerspectiveTransform, warpPerspective, line, circle, rectangle, perspectiveTransform
# 
matrixs = HomographyMatrix()


# u = np.array([745, 1720, 470, 2010])
# v = np.array([350, 350, 1070, 1060])

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

homoScenes = []

TL = court_keypoints_predictions[0][4]  # 4  5  
TR = court_keypoints_predictions[0][6]  # ? 6   7
LL = court_keypoints_predictions[0][5]  # ??  5   4
LR = court_keypoints_predictions[0][7]  # 7   6

homoScene, M = matrixs.courtMap(frames[0], *[TL, TR, LL, LR])   # [TL, TR, LL, LR]

homoScene = matrixs.showLines(homoScene)

new_scene = homoScene.copy()
#           MVP IS READY
# 
#   NEXT STEP: TRANSFORM PLAYERS AND BALL COORDINATES TO MINI MAP

#
class MiniMap:
    def __init__(self) -> None:
        self.coordinates = None
        
    # def convert(self, xy: List)-> List:
    #     """ Conver pixels to meters """
    #     # some code here
    #     assert isinstance(xy, List)
    #     return xy
    
    def calculate_homomatrix(self, frame):
        """ Calculated homography matrix and also return transformed image """
        pass
    
    def transform(self, frames):
        """ Homomatrix transform frames to horizontali point of view """
        # ... some code here
        assert isinstance(frames, List)
        return frames

# Init  
mini_mapper = MiniMap()
# Convert 2D img coordinates to the horizontal plane to mini_court positions
mini_ball_predictions = mini_mapper.transform(ball_predictions)
mini_players_predictions = mini_mapper.transform(ball_predictions)


from src.scripts import BounceDetector, SecondBouncer  # , BallHit, Score, StepCount # BallCheck  # , VideoRetiming, Stabilization

# Use Optical Flow for improving quality of n_frames ball's detections. The frames will be more slow motion
# Use Stabilization by court detections keypoints


# TODO: Try MVP  Ball Bounce and Ball Check  
#   TRACK BALL POSITION
ball_bouncer = BounceDetector('./models/ctb_regr_bounce.cbm')
ball_bouncer_1 = SecondBouncer()

ball_bounces = ball_bouncer.predict(array_ball_predictions[:, 0], array_ball_predictions[:, 1], smooth=True)   # Return list of indexcies
ball_bounces_1 = ball_bouncer_1.predict(ball_predictions)

mini_ball_bounces = mini_mapper.transform(ball_bounces)

print()
""" First idea: Calculate direction """
# ball_hitter = BallHit()
# score_counter = Score()
# step_counter = StepCount()
# 
# ball_checker = BallCheck()

# ball_hits = ball_hitter(ball_predictions)

# 
# mini_ball_hits = mini_mapper.transform(ball_hits)
# Ball Checker
# ball_is_in_out = ball_checker(mini_ball_predictions, mini_ball_bounces)

# self.model = Model


# TODO: 2. показывает сразу все позиции на мини-карте. DONE. Но не знаю как удалить старые, может сохранять по батчу

# Watch the result
output_video_path = "./results/output_homo.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, 
                     fourcc, 
                     24, 
                     (homoScene.shape[1], homoScene.shape[0])
                     )

center_players = []
# Plot players every players on the cort
for i, (players_coords, ball_coords) in enumerate(zip(player_predictions, ball_predictions)):
    for key, player_coord in players_coords.items():
        # 
        player_coord = get_center_of_bbox(player_coord)
        # 
        # homoScene = matrixs.showPoint(homoScene, M, player_coord)  # [800, 1130])    # homoScene = matrixs.showPoint(homoScene, M, [1400, 340])  # UP???
        points =  matrixs.givePoint(M, player_coord)
        new_scene = cv2.circle(homoScene, center=points, radius=10, color=(0, 0, 255))
    # 
    ball_center = np.float32([[ball_coords]]) # Transform to needed format
    transformed = perspectiveTransform(ball_center, M)[0][0]
    new_scene = circle(new_scene, (int(transformed[0]), int(transformed[1])), radius=5, color=(0, 255, 255), thickness=2)

    # Ball Bounce positions
    if i in ball_bounces:
        new_scene = circle(new_scene, (int(transformed[0]), int(transformed[1])), radius=15, color=(0, 0, 0), thickness=5)
    
    out.write(new_scene)
    # cv2.imwrite("./results/first_homo_img.jpg", new_scene)  # plt.

    homoScenes.append(new_scene)

out.release()


# save_video(homoScenes, output_video_path)




save_minimap = True

if save_minimap:
    pass

# minimap = cv2.resize(minimap, ())


"""
    На подумать:
      - BallBounce и BallHit и BallCheck - это массив из индексов
      - Модель для предсказания отскока справляется так себе, надо доучивать/переучивать. Для улучшения 2ого метода нужна более точная модель предсказания мяча
      - Попробовать всё таки трекать, где используем YOLO. Часто перекидывает на нежуные мячи. Решение: Попробовать трекать область с мячом, 
      например квадрат со сторонами 250 пикселей и идти скользящим окном по фото с небольшим отставанием от предыдущего 
      - Как лучше всего считать скорость
      """


class SpeedEstimator:
    def __init__(self) -> None:
        pass
    
    def calculate(self, ball_predections: List):
        assert type(ball_predections) == List
        pass
    pass


speed_estimator = SpeedEstimator()
ball_speed = speed_estimator.calculate(mini_ball_predictions)

#  APPLY BOTH LAST LAYERS ()

annotated_video = [mini_ball_predictions, mini_ball_hits, mini_ball_bounces, mini_players_predictions]

# Plot Mini Map info
# 


# Predict


# Visualize




# class Video:
#     frames: List[...]



# https://arxiv.org/ftp/arxiv/papers/2404/2404.06977.pdf
