# Directions of the ball

"""
    From https://github.com/yastrebksv/TennisProject/blob/main/bounce_detector.py#L4
"""
import catboost as ctb
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import distance

import warnings
warnings.filterwarnings("ignore")


class BounceDetector:
    def __init__(self, path_model=None):
        self.model = ctb.CatBoostRegressor()
        self.threshold = 0.45
        if path_model:
            self.load_model(path_model)
        
    def load_model(self, path_model):
        self.model.load_model(path_model)
    
    def prepare_features(self, x_ball, y_ball):
        labels = pd.DataFrame({'frame': range(len(x_ball)), 'x-coordinate': x_ball, 'y-coordinate': y_ball})
        
        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels['x_lag_{}'.format(i)] = labels['x-coordinate'].shift(i)
            labels['x_lag_inv_{}'.format(i)] = labels['x-coordinate'].shift(-i)
            labels['y_lag_{}'.format(i)] = labels['y-coordinate'].shift(i)
            labels['y_lag_inv_{}'.format(i)] = labels['y-coordinate'].shift(-i) 
            labels['x_diff_{}'.format(i)] = abs(labels['x_lag_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_{}'.format(i)] = labels['y_lag_{}'.format(i)] - labels['y-coordinate']
            labels['x_diff_inv_{}'.format(i)] = abs(labels['x_lag_inv_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_inv_{}'.format(i)] = labels['y_lag_inv_{}'.format(i)] - labels['y-coordinate']
            labels['x_div_{}'.format(i)] = abs(labels['x_diff_{}'.format(i)]/(labels['x_diff_inv_{}'.format(i)] + eps))
            labels['y_div_{}'.format(i)] = labels['y_diff_{}'.format(i)]/(labels['y_diff_inv_{}'.format(i)] + eps)

        for i in range(1, num):
            labels = labels[labels['x_lag_{}'.format(i)].notna()]
            labels = labels[labels['x_lag_inv_{}'.format(i)].notna()]
        labels = labels[labels['x-coordinate'].notna()] 
        
        colnames_x = ['x_diff_{}'.format(i) for i in range(1, num)] + \
                     ['x_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['x_div_{}'.format(i) for i in range(1, num)]
        colnames_y = ['y_diff_{}'.format(i) for i in range(1, num)] + \
                     ['y_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['y_div_{}'.format(i) for i in range(1, num)]
        colnames = colnames_x + colnames_y

        features = labels[colnames]
        return features, list(labels['frame'])
    
    def predict(self, x_ball, y_ball, smooth=True):
        """"""
        # x_ball, y_ball = ball_detection

        if smooth:
            x_ball, y_ball = self.smooth_predictions(x_ball, y_ball)
        features, num_frames = self.prepare_features(x_ball, y_ball)
        preds = self.model.predict(features)
        ind_bounce = np.where(preds > self.threshold)[0]
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess(ind_bounce, preds)
        frames_bounce = [num_frames[x] for x in ind_bounce]
        return frames_bounce
    
    def smooth_predictions(self, x_ball, y_ball):
        is_none = [int(x is None) for x in x_ball]
        interp = 5
        counter = 0
        for num in range(interp, len(x_ball)-1):
            if not x_ball[num] and sum(is_none[num-interp:num]) == 0 and counter < 3:
                x_ext, y_ext = self.extrapolate(x_ball[num-interp:num], y_ball[num-interp:num])
                x_ball[num] = x_ext
                y_ball[num] = y_ext
                is_none[num] = 0
                if x_ball[num+1]:
                    dist = distance.euclidean((x_ext, y_ext), (x_ball[num+1], y_ball[num+1]))
                    if dist > 80:
                        x_ball[num+1], y_ball[num+1], is_none[num+1] = None, None, 1
                counter += 1
            else:
                counter = 0  
        return x_ball, y_ball

    def extrapolate(self, x_coords, y_coords):
        xs = list(range(len(x_coords)))
        func_x = CubicSpline(xs, x_coords, bc_type='natural')
        x_ext = func_x(len(x_coords))
        func_y = CubicSpline(xs, y_coords, bc_type='natural')
        y_ext = func_y(len(x_coords))
        return float(x_ext), float(y_ext)    

    def postprocess(self, ind_bounce, preds):
        ind_bounce_filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i-1]) != 1:
                cur_ind = ind_bounce[i]
                ind_bounce_filtered.append(cur_ind)
            elif preds[ind_bounce[i]] > preds[ind_bounce[i-1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
        return ind_bounce_filtered
    

class SecondBouncer:
    def predict(self, ball_detections):
        # convert the list into pandas.DataFrame
        df_ball_positions = pd.DataFrame(np.array(ball_detections), columns=['x', 'y'], dtype=float)
        # interpolate the missing values
        print(df_ball_positions.isna().sum())
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # MA 
        df_ball_positions['y_ma'] = df_ball_positions['y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['x_ma'] = df_ball_positions['x'].rolling(window=5, min_periods=1, center=False).mean()
        # 
        df_ball_positions['delta_y'] = df_ball_positions['y_ma'].diff()

        df_ball_positions['ball_hit'] = 0

        # 
        minimum_change_frames_for_hit = 25

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

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


        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()
        # df_ball_positions[df_ball_positions['ball_hit']==1]

        return frame_nums_with_ball_hits

