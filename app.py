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
path = r'data/vizualize/tennis_1.mp4'
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


""" SAVE VIDEO """
# output_video_1 = frames.copy()
# output_video_1 = FieldsKeypointsDetection.draw_keypoints(output_video_1, court_keypoints_predictions)
# output_video_1 = ball_detectior.draw_bboxes(video_frames=output_video_1, ball_detections=ball_predictions)
# output_video_path = "./results/output_1.mp4"
# save_video(output_video_1, output_video_path)


# 1. Use other model, for Shot Recognition and Predicting missing points
# Predict missing points in ball detections
from src.scripts import FillBallDetections, ShotType

# Concat ball detections and Type of shoе predict (classification)
# results


shot_recognitor = ShotType()  # Use predicted Players coordinates of body keypoints
types_of_shot = shot_recognitor(player_predictions)

assert len(ball_predictions) == len(frames)

ball_filler = FillBallDetections()
ball_predictions = ball_filler.choose_main_ball(ball_predictions)
assert len(ball_predictions) == len(frames)
ball_predictions = ball_filler.interpolate_ball_position(ball_predictions) 

assert len(ball_predictions) == len(frames)
# TODO: 1. Научится выбирать мяч
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

homoFrames = []

TL = court_keypoints_predictions[0][4]
TR = court_keypoints_predictions[0][6]  # ?
LL = court_keypoints_predictions[0][5]  # ??
LR = court_keypoints_predictions[0][7]

homoFrame, M = matrixs.courtMap(frames[0], *[TL, TR, LL, LR])   # [TL, TR, LL, LR]

homoFrame = matrixs.showLines(homoFrame)


# TODO: 2. показывает сразу все позиции на мини-карте
# Find Center of bbox for every players
center_players = []
# Plot players every players on the cort
for players_coords, ball_coords in zip(player_predictions, ball_predictions):
    for key, player_coord in players_coords.items():
        # 
        player_coord = get_center_of_bbox(player_coord)
        # 
        # homoFrame = matrixs.showPoint(homoFrame, M, player_coord)  # [800, 1130])    # homoFrame = matrixs.showPoint(homoFrame, M, [1400, 340])  # UP???
        points =  matrixs.givePoint(M, player_coord)
        cv2.circle(homoFrame, center=points, radius=1, color=(0, 0, 255))

    # 
    # ball_center = get_center_of_bbox(ball_coords)
    ball_center = np.float32([[ball_coords]]) # Transform to needed format
    transformed = perspectiveTransform(ball_center, M)[0][0]
    circle(homoFrame, (int(transformed[0]), int(transformed[1])), radius=0, color=(0, 255, 255), thickness=25)
        

    homoFrames.append(homoFrame)

# Watch the result
output_video_path = "./results/output_homo.mp4"
save_video(homoFrames, output_video_path)



# 
#   NEXT STEP: TRANSFORM PLAYERS AND BALL COORDINATES TO MINI MAP
# 
class MiniMap:
    def __init__(self) -> None:
        self.coordinates = None
        
    def convert(self, frames: List)-> List:
        """ Conver pixels to meters """
        # some code here
        return frames
    
    def transform(self, frames):
        # some code here
        assert isinstance(frames, List)
        return frames

# Init  
mini_mapper = MiniMap()
# Convert 2D img coordinates to the horizontal plane to mini_court positions
mini_ball_predictions = mini_mapper.transform(ball_predictions)
mini_players_predictions = mini_mapper.transform(ball_predictions)


from src.scripts import BallBounce, BallHit, Score, StepCount, BallCheck  # , VideoRetiming, Stabilization

# Use Optical Flow for improving quality of n_frames ball's detections. The frames will be more slow motion
# Use Stabilization by court detections keypoints


#   TRACK BALL POSITION
ball_bouncer = BallBounce()
ball_hitter = BallHit()
# score_counter = Score()
# step_counter = StepCount()
# 
ball_checker = BallCheck()

ball_bounces = ball_bouncer(ball_predictions)
mini_ball_bounces = mini_mapper.transform(ball_bounces)

ball_hits = ball_hitter(ball_predictions)
# 
mini_ball_hits = mini_mapper.transform(ball_hits)
# Ball Checker
ball_is_in_out = ball_checker(mini_ball_predictions, mini_ball_bounces)

# self.model = Model
"""
    На подумать:
      - BallBounce и BallHit и BallCheck - это массив из 0 и 1, или лучше массив из индексов?
      """


class SpeedEstimator:
    def __init__(self) -> None:
        pass
    
    def calculate(self, ball_predections: List):
        assert type(frames) == List

        # rames = 

        # read the picle file
        with open('../tracker_stubs/ball_detections.pkl', 'rb') as f:
            ball_positions = pickle.load(f)

        df_ball_positions = pd.DataFrame()

        ball_positions = [x.get(1, []) for x in ball_positions]   # if not detections return []
        # convert the list into pandas.DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Convert pandas.DataFrame to original format. Back up
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()

        df_ball_positions['mid_x'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['mid_x_rolling_mean'] = df_ball_positions['mid_x'].rolling(window=5, min_periods=1, center=False).mean()

        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()


        df_ball_positions['ball_hit'] = 0

        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

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
