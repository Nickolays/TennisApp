# 
# 1. Ball in or Out
#   1.1 Find indecies, where every shots were.
#   1.2 Coordinates of image to hit map coords in the image
#   1.3 Compare homograpthed coordinates of ball with Main coordinates 

# 2. Speed 
#   2.1 Find indecies, where every shots were.
#   2.2 Coordinates of image to hit map coords in the image
#   2.3 Use moving average, nearby BallHit indicies

# 3. Hit Map 
#   3.1 Find indecies, where every shots were.
#   3.2 Convert coordinates of image to hit map coords in the image 
#   3.3 Plot into image point of hit map, where shot was

#


class HomographyMatrix:
    def __init__(self) -> None:
        """
            Every n frames we update our homorgraphy matrix
        """
        # self.matrixs = {}
        pass

    def create(self):
        pass

    def update(self):

        pass

    def calculate(self):
        pass

    def __call__(self, frames: List, court_detections: List, n_skip_frames: int) -> List:
        """
            frames - Images into List
            n_skip_frames - How many frames we skipped to obtain new homo-matrix

          return:
            - None of Homographic Matrix
        
        """
        # Init first matrix
        # self.matrixs = self.create(frames[0])
        # Here we save 
        outputs = [] 
        for indx in range(1, len(frames)):
            # We have at least 4 keypoints
            # if len(court_detections[indx]) > 3:




            if indx // n_skip_frames:
                outputs.append(
                    self.update(frames[indx])
                )
            else:
                outputs.append(None)

        return outputs
            
        # pass



import cv2
import numpy as np
from ultralytics import YOLO
classes_names_dic

video_path = r'/home/suetin/Projects/VSCode/TennisApp/SimpleApp/input_videos/tennis_1.mp4'
p_model_path = r'/home/suetin/Projects/VSCode/TennisApp/SimpleApp/models/yolov8x.pt'
kp_model_path = r'/home/suetin/Projects/VSCode/TennisApp/SimpleApp/models/model_tennis_court_det.pt'

# Read tactical map image
tac_map = cv2.imread('./tactical map.jpg')

from sklearn.metrics import mean_squared_error

# Load the YOLOv8 players detection model
model_players = YOLO(p_model_path)

# Load the YOLOv8 field keypoints detection model
model_keypoints = YOLO(kp_model_path)


# Open video file
cap = cv2.VideoCapture(video_path)

# Initialize frame counter
frame_nbr = 0

# Set keypoints average displacement tolerance level (in pixels) [set to -1 to always update homography matrix]
keypoints_displacement_mean_tol = 10

# Set confidence thresholds for players and field keypoints detections
player_model_conf_thresh = 0.60
keypoints_model_conf_thresh = 0.70

# Set variable to record the time when we processed last frame 
prev_frame_time = 0
# Set variable to record the time at which we processed current frame 
new_frame_time = 0

# Store the ball track history
ball_track_history = {'src':[],
                      'dst':[]
}

# Count consecutive frames with no ball detected
nbr_frames_no_ball = 0
# Threshold for number of frames with no ball to reset ball track (frames)
nbr_frames_no_ball_thresh = 30
# Distance threshold for ball tracking (pixels)
ball_track_dist_thresh = 100
# Maximum ball track length (detections)
max_track_length = 35

# Loop through the video frames
while cap.isOpened():

    # Update frame counter
    frame_nbr += 1

    # Read a frame from the video
    success, frame = cap.read()

    # Reset tactical map image for each new frame
    tac_map_copy = tac_map.copy()

    # Reset ball tracks
    if nbr_frames_no_ball>nbr_frames_no_ball_thresh:
            ball_track_history['dst'] = []
            ball_track_history['src'] = []

    # Process the frame if it was successfuly read
    if success:
        
        #################### Part 1 ####################
        # Object Detection & Coordiante Transofrmation #
        ################################################

        # Run YOLOv8 players inference on the frame
        results_players = model_players(frame, conf=player_model_conf_thresh)
        # Run YOLOv8 field keypoints inference on the frame
        results_keypoints = model_keypoints(frame, conf=keypoints_model_conf_thresh)

        ## Extract detections information
        bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()                          # Detected players, referees and ball (x,y,x,y) bounding boxes
        bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy()                        # Detected players, referees and ball (x,y,w,h) bounding boxes    
        labels_p = list(results_players[0].boxes.cls.cpu().numpy())                     # Detected players, referees and ball labels list
        confs_p = list(results_players[0].boxes.conf.cpu().numpy())                     # Detected players, referees and ball confidence level
        
        bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()                        # Detected field keypoints (x,y,w,h) bounding boxes
        bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()                        # Detected field keypoints (x,y,w,h) bounding boxes
        labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())                   # Detected field keypoints labels list

        # Convert detected numerical labels to alphabetical labels
        detected_labels = [classes_names_dic[i] for i in labels_k]

        # Extract detected field keypoints coordiantes on the current frame
        detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

        # Get the detected field keypoints coordinates on the tactical map
        detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])


        ## Calculate Homography transformation matrix when more than 4 keypoints are detected
        if len(detected_labels) > 3:
            # Always calculate homography matrix on the first frame
            if frame_nbr > 1:
                # Determine common detected field keypoints between previous and current frames
                common_labels = set(detected_labels_prev) & set(detected_labels)
                # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
                if len(common_labels) > 3:
                    common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]   # Get labels indexes of common detected keypoints from previous frame
                    common_label_idx_curr = [detected_labels.index(i) for i in common_labels]        # Get labels indexes of common detected keypoints from current frame
                    coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]     # Get labels coordiantes of common detected keypoints from previous frame
                    coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]          # Get labels coordiantes of common detected keypoints from current frame
                    coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  # Calculate error between previous and current common keypoints coordinates
                    update_homography = coor_error > keypoints_displacement_mean_tol                 # Check if error surpassed the predefined tolerance level
                else:
                    update_homography = True                                                         
            else:
                update_homography = True

            if  update_homography:
                h, mask = cv2.findHomography(detected_labels_src_pts,                   # Calculate homography matrix
                                              detected_labels_dst_pts)                  
            
            detected_labels_prev = detected_labels.copy()                               # Save current detected keypoint labels for next frame
            detected_labels_src_pts_prev = detected_labels_src_pts.copy()               # Save current detected keypoint coordiantes for next frame

            bboxes_p_c_0 = bboxes_p_c[[i==0 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected players (label 0)
            bboxes_p_c_2 = bboxes_p_c[[i==2 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected ball(s) (label 2)

            # Get coordinates of detected players on frame (x_cencter, y_center+h/2)
            detected_ppos_src_pts = bboxes_p_c_0[:,:2]  + np.array([[0]*bboxes_p_c_0.shape[0], bboxes_p_c_0[:,3]/2]).transpose()
            # Get coordinates of the first detected ball (x_center, y_center)
            detected_ball_src_pos = bboxes_p_c_2[0,:2] if bboxes_p_c_2.shape[0]>0 else None

            # Transform players coordinates from frame plane to tactical map plance using the calculated Homography matrix
            pred_dst_pts = []                                                           # Initialize players tactical map coordiantes list
            for pt in detected_ppos_src_pts:                                            # Loop over players frame coordiantes
                pt = np.append(np.array(pt), np.array([1]), axis=0)                     # Covert to homogeneous coordiantes
                dest_point = np.matmul(h, np.transpose(pt))                              # Apply homography transofrmation
                dest_point = dest_point/dest_point[2]                                   # Revert to 2D-coordiantes
                pred_dst_pts.append(list(np.transpose(dest_point)[:2]))                 # Update players tactical map coordiantes list
            pred_dst_pts = np.array(pred_dst_pts)

            # Transform ball coordinates from frame plane to tactical map plane using the calculated Homography matrix
            if detected_ball_src_pos is not None:
                pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                dest_point = np.matmul(h, np.transpose(pt))
                dest_point = dest_point/dest_point[2]
                detected_ball_dst_pos = np.transpose(dest_point)

                # Update track ball position history
                if len(ball_track_history['src'])>0 :
                    if np.linalg.norm(detected_ball_src_pos-ball_track_history['src'][-1])<ball_track_dist_thresh:
                        ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                        ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                    else:
                        ball_track_history['src']=[(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                        ball_track_history['dst']=[(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
                else:
                    ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                    ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
            # Remove oldest tracked ball postion if track exceedes threshold        
            if len(ball_track_history) > max_track_length:
                    ball_track_history['src'].pop(0)
                    ball_track_history['dst'].pop(0)