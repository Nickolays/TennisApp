from typing import List
import pandas as pd
import numpy as np


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1, p2):
    """
    p1 - player_center
    p2 - court_keypoint
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


class FillBallDetections:
    """ 
    Simple idea is use Classical ML Regression model. In the future try RNN model 
    """
    def __init__(self) -> None:

        self.min_pix_dist = 80

    def interpolate_ball_position(self, df_ball_positions):
        """  """
        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate(method='linear', axis=0)
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.round().astype('int')

        return df_ball_positions
    
    def preprocess(self, ball_detections, window=5):
        """  """
        # convert the list into pandas.DataFrame
        df_ball_positions = pd.DataFrame(ball_detections, columns=['x', 'y'])

        # Calc features
        df_ball_positions['y_ma'] = df_ball_positions['y'].rolling(window=window, min_periods=1, center=False).mean()
        df_ball_positions['x_ma'] = df_ball_positions['x'].rolling(window=window, min_periods=1, center=False).mean()

        df_ball_positions['delta_y'] = df_ball_positions['y_ma'].diff()
        df_ball_positions['delta_x'] = df_ball_positions['x_ma'].diff() 

        df_ball_positions['y_ma_shifted'] = df_ball_positions['y_ma'].shift(-1)
        df_ball_positions['x_ma_shifted'] = df_ball_positions['x_ma'].shift(1)
        df_ball_positions = df_ball_positions.fillna(0)
        # Calc distance
        df_ball_positions['distance'] = ((df_ball_positions['x_ma'] - df_ball_positions['x_ma_shifted'])**2 \
                                + (df_ball_positions['y_ma'] - df_ball_positions['y_ma_shifted'])**2) 
        # Choose main ball and replace to Nan extreme values
        extreme_value = df_ball_positions['distance'][1:].quantile(0.95)
        df_ball_positions['x'] = np.where(df_ball_positions['distance'] < extreme_value, df_ball_positions['x'].values, np.NaN)
        df_ball_positions['y'] = np.where(df_ball_positions['distance'] < extreme_value, df_ball_positions['y'].values, np.NaN)

        # Interpolate Nans
        df_ball_positions = self.interpolate_ball_position(df_ball_positions)
        
        # Convert to original format
        ball_positions = df_ball_positions[['x', 'y']].values.tolist()
        return ball_positions
    
    @staticmethod
    def get_center_of_bbox(bbox):
        try:
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            return (center_x, center_y)
        except:
            print("NAN HERE")
            return (None, None)

    @staticmethod
    def measure_distance(p1, p2):
        """
        p1 - player_center
        p2 - court_keypoint
        """
        try:
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        except:
            return 999
    
    def choose_main_ball(self, ball_list):
        """     IT DOESN'T WORK 
            Calculation distance between shifted ball positions using Pandas
          1. Shift data to 1 step
          2. Calc difference
          3. Transform to distance
          4. Choose ball with minimal distance between previous frame and new frame
          5. Fill wrong balls Nan
        """
        # Take first array and apply transfrom for this frame
        new_ball_list = []
        new_ball_list.append([None, None])
        # Calc mean coordinates for ball
        last_good_coords= pd.DataFrame(ball_list).mean().to_list()
        # 
        for i in range(len(ball_list[1:])):
            # Pixel distance
            # distance = self.measure_distance(ball_list[i+1], ball_list[i])
            distance = self.measure_distance(ball_list[i+1], last_good_coords)
            if distance < self.min_pix_dist:
                new_ball_list.append(ball_list[i+1])
                last_good_coords = ball_list[i+1]
            elif distance == 999:
                new_ball_list.append([None, None])
            else:
                new_ball_list.append([None, None])
                # last_good_coords = ball_list[i]
        
        print(len(new_ball_list[-1]), len(ball_list[-1]))
        print(len(new_ball_list), len(ball_list))
        assert len(new_ball_list) == len(ball_list)
        return new_ball_list

        # distances = []
        # for bbox in ball_list:
        #     player_center = get_center_of_bbox(bbox)

        #     min_distance = float('inf')
        #     # for i in range(0,len(court_keypoints), 2):
        #     for court_keypoint in court_keypoints:
        #         #court_keypoint = (court_keypoints[i], court_keypoints[i+1])
        #         # print(player_center)
        #         # print()
        #         # print(f"Court kpoints: {court_keypoint}")
        #         # print()
        #         distance = measure_distance(player_center, court_keypoint)
        #         distance = distance.cpu().numpy()
        #         # print(f"Distance : {distance}") # Here is array
        #         # print()
        #         # print(min_distance)   # inf
        #         if distance < min_distance:
        #             min_distance = distance
        #     distances.append(min_distance)
        
        # # sorrt the distances in ascending order
        # distances.sort(key = lambda x: x[1])
        # # Choose the first 2 tracks
        # chosen_players = [distances[0][0], distances[1][0]]
        # return chosen_players

        return coordinates

    def __call__(self, ball_detections)-> List:
        # some code ...
        assert isinstance(ball_detections, List)

        output = self.interpolate_ball_position(ball_detections)

        assert len(output) == len(ball_detections)
        return ball_detections



